import os
import csv
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog, QSplitter, QComboBox, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from load_and_predict import  predicto, ImagePreprocessor
from image_viewer import ImageViewer
from ui_components import create_left_layout, create_right_widget


class MelanomaDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.models = {
            'DenseNet': 'best_model_DenseNet121.h5',
            'ResNet': 'best_model_ResNet50.h5',
            'Xception': 'best_model_Xception.h5',
            'MobileNet': 'best_model_MobileNet.h5',
            'Inception': 'best_model_InceptionV3.h5',
            'EfficientNet': 'best_model_EfficientNetV2B0.h5'
        }
        self.current_model = None
        self.preprocessor = ImagePreprocessor()
        self.image_history = []
        self.current_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Detector de Melanoma Avanzado')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        left_layout = create_left_layout(self)
        right_widget = create_right_widget(self)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_widget)

        # Actualiza la lista de modelos en el selector
        #self.model_selector.addItems(self.models.keys())
        #self.model_selector.currentIndexChanged.connect(self.load_model)

    """def load_model(self):
        selected_model = self.model_selector.currentText()
        model_path = self.models[selected_model]
        self.current_model = load_trained_model(model_path)"""

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de Imagen (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_viewer.set_image(pixmap)
            self.current_image = file_name
            self.image_history.append(file_name)
            self.history_list.addItem(os.path.basename(file_name))

              # Limpia los resultados anteriores
        self.result_text.clear()
        self.comparison_text.clear()

    def load_from_history(self, item):
        file_name = self.image_history[self.history_list.row(item)]
        pixmap = QPixmap(file_name)
        self.image_viewer.set_image(pixmap)
        self.current_image = file_name

    def analyze_image(self):
        if not self.current_image:
            self.result_text.setText("Por favor, cargue una imagen primero.")
            return

        """if not self.current_model:
            self.result_text.setText("Por favor, seleccione un modelo primero.")
            return"""

        patient_data = {
            "Nombre": self.name_input.text(),
            "Identificación": self.id_input.text(),
            "Edad": self.age_input.text(),
            "Sexo": self.sex_input.currentText(),
            "Localización": self.location_input.currentText()
        }
        # Verifica que la imagen cargada es la correcta
        print(f"Analizando imagen: {self.current_image}")
        result = predicto(self.current_model, self.current_image, patient_data["Edad"], patient_data["Sexo"], patient_data["Localización"])

        # Construye el texto para mostrar los resultados de la predicción
        result_text = f"Análisis completado para {patient_data['Nombre']} (ID: {patient_data['Identificación']}).\n"
        result_text += f"Edad: {patient_data['Edad']}, Sexo: {patient_data['Sexo']}, Localización: {patient_data['Localización']}\n\n"
        result_text += f"Clase predicha: {result['predicted_class']}\n"
        result_text += "Probabilidades:\n"
        
        for class_name, probability in result['probabilities'].items():
            result_text += f"{class_name}: {probability:.4f}\n"

        result_text += "\n" + ("Se recomienda consultar a un dermatólogo." if result['probabilities']['mel'] > 0.5 else "El riesgo parece bajo, pero consulte a un médico si tiene dudas.")

        self.result_text.setText(result_text)

        self.save_to_csv(patient_data, result['probabilities'])


    def save_to_csv(self, patient_data, probabilities):
        csv_file = 'historial_pacientes.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['Fecha', 'Nombre', 'Identificación', 'Edad', 'Sexo', 'Localización', 'Imagen', 'Probabilidades']
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
                'Probabilidades': str(probabilities)
            })


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
