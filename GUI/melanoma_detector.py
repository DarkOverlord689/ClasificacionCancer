"""
    Este mpodulo posee las funciones pricipales que se ejecutan en las interfaz gráfica
    Este hace una importación del módulo de componentes
    Esta diseñado con una clase principal y funciones 

"""

# ------------------------ IMportación de librerias--------------------------------
import os
import csv
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog,  QLabel
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from load_and_predict import  predicto, ImagePreprocessor
from ui_components import create_left_layout, create_right_widget
from PyQt6.QtWidgets import QTableWidgetItem



# Clase principal MelanomaDetector que hereda de QMainWindow para gestionar la ventana principal de la GUI.


class MelanomaDetector(QMainWindow):
    def __init__(self):
         # Constructor de la clase, inicializa variables y configura la interfaz
        super().__init__()
        # se crea un diccionario que mapea nombres de modelos a los archivos correspondientes
        self.models = {
            'DenseNet': 'best_model_DenseNet121.h5',
            'ResNet': 'best_model_ResNet50.h5',
            'Xception': 'best_model_Xception.h5',
            'MobileNet': 'best_model_MobileNet.h5',
            'Inception': 'best_model_InceptionV3.h5',
            'EfficientNet': 'best_model_EfficientNetV2B0.h5'
        }

        # Variables para el modelo actual, preprocesador de imágenes e historial de imágenes
        self.current_model = None
        self.preprocessor = ImagePreprocessor()
        self.image_history = []
        self.current_image = None
        # Diccionario para traducir abreviaciones de tipos de cáncer a nombres completos
        self.cancer_types = {
            'mel': 'Melanoma',
            'nv': 'Nevo',
            'bcc': 'Carcinoma Basocelular',
            'akiec': 'Queratosis Actínica',
            'bkl': 'Queratosis Benigna',
            'df': 'Dermatofibroma',
        }
        # Inicializa la interfaz gráfica
        self.initUI()


    # Función para actualizar el nivel de zoom en la imagen
    def update_zoom(self, value):
        if hasattr(self, 'image_viewer'):   
            self.image_viewer.update_zoom(value)

    # Función para actualizar el contraste en la imagen
    def update_contrast(self, value):
        if hasattr(self, 'image_viewer'):
            self.image_viewer.update_contrast(value)

    # Función para actualizar el brillo en la imagen
    def update_brightness(self, value):
        if hasattr(self, 'image_viewer'):
            self.image_viewer.update_brightness(value)

    # Configuración inicial de la interfaz gráfica de usuario (UI)
    def initUI(self):

        # Configura el título de la ventana y su tamaño
        self.setWindowTitle('Dermadetect')
        self.setGeometry(100, 100, 1200, 800)


        # Crea el widget central y el layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Crea los layouts izquierdo y derecho para la interfaz utilizando funciones personalizadas

        left_layout = create_left_layout(self) #componente izquierdo
        right_widget = create_right_widget(self) #componente derecho

        # Agrega ambos layouts al layout principa
        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_widget)

        
        # Actualiza la lista de modelos en el selector
        #self.model_selector.addItems(self.models.keys())
        #self.model_selector.currentIndexChanged.connect(self.load_model)

    """def load_model(self):
        selected_model = self.model_selector.currentText()
        model_path = self.models[selected_model]
        self.current_model = load_trained_model(model_path)"""


    # Función para cargar una imagen desde el sistema de archivos
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de Imagen (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_viewer.set_image(pixmap)
            self.current_image = file_name
            self.image_history.append(file_name)

            # Guardar el nombre y la probabilidad en la tabla de historial
           # self.add_to_history_table(os.path.basename(file_name), self.predicted_class, self.prediction_probability)

            # Limpia los resultados anteriores
            self.result_text.clear()
            #self.comparison_text.clear()

    """def add_to_history_table(self, file_name, predicted_class, probability):
        row_position = self.history_table.rowCount()
        self.history_table.insertRow(row_position)
        self.history_table.setItem(row_position, 0, QTableWidgetItem(file_name))
        self.history_table.setItem(row_position, 1, QTableWidgetItem(predicted_class))
        self.history_table.setItem(row_position, 2, QTableWidgetItem(f"{probability:.2f}%"))
    """
     # Función para cargar una imagen desde el historial
    def load_from_history(self, item):
        row = item.row()  # Obtiene la fila seleccionada
        file_name = self.history_table.item(row, 0).text()  # Obtiene el nombre de la imagen de la tabla
        pixmap = QPixmap(self.image_history[row])  # Carga la imagen del historial
        self.image_viewer.set_image(pixmap)  # La muestra en el visor
        self.current_image = file_name  # Actualiza la imagen actual


    # Actualiza la pestaña de comparación con imágenes y descripciones
    def update_comparison_tab(self, images_and_descriptions):
        # Limpiar el layout actual
        while self.comparison_layout.count():
            item = self.comparison_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # Añadir nuevas imágenes y descripciones
        for index, (image_file, description) in enumerate(images_and_descriptions):
            row = index // 2  # Calcula la fila (0, 1, 2)
            col = index % 2   # Calcula la columna (0 o 1)
            
            # Cargar y mostrar la imagen
            image_label = QLabel()
            pixmap = QPixmap(image_file)
            if not pixmap.isNull():
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                image_label.setText("Imagen no encontrada")
            self.comparison_layout.addWidget(image_label, row * 2, col, alignment=Qt.AlignmentFlag.AlignCenter)
            
            # Añadir la descripción debajo de la imagen
            description_label = QLabel(description)
            description_label.setFont(QFont("Arial", 12))
            description_label.setWordWrap(True)
            self.comparison_layout.addWidget(description_label, row * 2 + 1, col, alignment=Qt.AlignmentFlag.AlignCenter)

    """
    Esta es la funcion mas importante, aqui se realiza la predicción del modelo
    analiza la imagen actual y muestra los resultados
    """
    def analyze_image(self):
        if not self.current_image:
            self.result_text.setText("<p style='color: red;'>Por favor, cargue una imagen primero.</p>")
            return

        # Recolecta los datos del paciente
        patient_data = {
            "Nombre": self.name_input.text(),
            "Identificación": self.id_input.text(),
            "Edad": self.age_input.text(),
            "Sexo": self.sex_input.currentText(),
            "Localización": self.location_input.currentText()
        }
        # Llama a la función de predicción del modelo seleccionado
        result = predicto(self.current_model, self.current_image, patient_data["Edad"], patient_data["Sexo"], patient_data["Localización"])

        # Convertir abreviaciones a nombres completos
        full_class_name = self.cancer_types.get(result['predicted_class'], result['predicted_class'])
         # Muestra los resultados predichos y las probabilidadesx|
        result_text = f"""
        <h3>Resultados del Análisis</h3>
        <p><b>Nombre:</b> {patient_data['Nombre']} <br>
        <b>Identificación:</b> {patient_data['Identificación']} <br>
        <b>Edad:</b> {patient_data['Edad']} <br>
        <b>Sexo:</b> {patient_data['Sexo']} <br>
        <b>Localización:</b> {patient_data['Localización']}</p>
        <p><b>Clase predicha:</b> {full_class_name}</p>
        <h4>Probabilidades:</h4>
        <ul>
        """

         # Lista las probabilidades para cada clase
        for class_name, probability in result['probabilities'].items():
            full_name = self.cancer_types.get(class_name, class_name)
            result_text += f"<li><b>{full_name}:</b> {probability:.4f}</li>"

        result_text += "</ul>"


        # Recomendación según la probabilidad más alta
        max_probability = max(result['probabilities'].values())

        if max_probability > 0.5:
            recommendation = "<p style='color: red;'><b>Se recomienda consultar a un dermatólogo.</b></p>"
        else:
            recommendation = "<p style='color: green;'><b>El riesgo parece bajo, pero consulte a un médico si tiene dudas.</b></p>"

        result_text += recommendation

        # Actualiza la pestaña de resultados con el texto generado
        self.result_text.setHtml(result_text)

        # Guarda los resultados en un archivo CSV
        self.save_to_csv(patient_data, max_probability, full_class_name)

        
        # Actualiza la pestaña de comparación con imágenes de diferentes modelos
        images_and_descriptions = [
            ('interpretation_DenseNet121.jpg', 'Modelo DenseNet21'),
            ('interpretation_EfficientNetV2B0.jpg', 'Modelo EfficientNet'),
            ('interpretation_InceptionV3.jpg', 'Modelo Inception'),
            ('interpretation_MobileNet.jpg', 'Modelo MobileNet'),
            ('interpretation_ResNet50.jpg', 'Modelo ResNet50'),
            ('interpretation_Xception.jpg', 'Modelo Xeption')
        ]

        # Actualizar la pestaña de comparación
        self.update_comparison_tab(images_and_descriptions)


    # Función para guardar los resultados en un archivo CSV
    def save_to_csv(self, patient_data, probabilities, predicted_class):
        csv_file = 'historial_pacientes.csv'
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['Fecha', 'Nombre', 'Identificación', 'Edad', 'Sexo', 'Localización', 'Imagen', 'Clase Predicha', 'Probabilidades']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Nombre': patient_data['Nombre'],
                'Identificación': patient_data['Identificación'],
                'Edad': patient_data['Edad'],
                'Sexo': patient_data['Sexo'],
                'Localización': patient_data['Localización'],
                'Imagen': self.current_image,
                'Clase Predicha': predicted_class,
                'Probabilidades': str(probabilities)
            })

        self.update_patient_history_table()

    # Función que actuaiza la pestaña pasicentes
    def update_patient_history_table(self):
        self.history_table.setRowCount(0)  # Limpiar la tabla

        csv_file = 'historial_pacientes.csv'
        if os.path.isfile(csv_file):
            with open(csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    row_position = self.history_table.rowCount()
                    self.history_table.insertRow(row_position)
                    for i, (key, value) in enumerate(row.items()):
                        self.history_table.setItem(row_position, i, QTableWidgetItem(value))


    def load_patient_history(self):
        csv_file = 'historial_pacientes.csv'
        if not os.path.isfile(csv_file):
            self.result_text.setText("No se encontró historial de pacientes.")
            return

        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            history = ""
            for row in reader:
                history += f"Fecha: {row['Fecha']}\n"
                history += f"Nombre: {row['Nombre']} (ID: {row['Identificación']})\n"
                history += f"Edad: {row['Edad']}, Sexo: {row['Sexo']}, Localización: {row['Localización']}\n"
                history += f"Probabilidad: {row['Probabilidad']}%\n"
                history += f"Recomendación: {row['Recomendación']}\n\n"

            self.result_text.setText(history)

    #guarda los resultados en un txt
    def save_results(self):
        if not self.current_image:
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Guardar Resultados", "", "Archivos de Texto (*.txt)")
        if file_name:
            with open(file_name, 'w') as f:
                f.write(f"Resultados para la imagen: {self.current_image}\n\n")
                f.write(self.result_text.toPlainText() + "\n\n")
                f.write("Comparación:\n")
                f.write(self.comparison_text.toPlainText())

if __name__ == "__main__":
    app = QApplication([])
    window = MelanomaDetector()
    window.show()
    app.exec()
