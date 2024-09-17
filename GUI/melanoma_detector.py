"""
    Este mpodulo posee las funciones pricipales que se ejecutan en las interfaz gráfica
    Este hace una importación del módulo de componentes
    Esta diseñado con una clase principal y funciones 

"""

import os
import csv
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QLabel, QTableWidgetItem, QTableWidget, QApplication, QGridLayout, QTabWidget
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, pyqtSlot
from load_and_predict import predicto, ImagePreprocessor
from ui_components import create_main_layout
from pdfgenerator import generate_pdf_report
from PyQt6.QtCore import QFileInfo
import pandas as pd
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QFrame, QWidget, QGridLayout
from PyQt6.QtGui import QPixmap, QFont, QWheelEvent
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QScrollArea, QSizePolicy
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QMessageBox

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(1, 1)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.1
        if event.angleDelta().y() < 0:
            factor = 1 / factor
        self.scale(factor, factor)


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
        self.cancer_types = {
            'melanoma': 'Melanoma',
            'basal cell carcinoma': 'Basocelular',
            'squamous cell carcinoma': 'Escamocelular',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dermadetect')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Crear el QTabWidget principal
        self.main_tab_widget = QTabWidget()
        main_layout.addWidget(self.main_tab_widget)

        # Crear las pestañas principales
        self.history_tab = QWidget()
        self.results_tab = QWidget()
        self.comparison_tab = QWidget()

        # Añadir las pestañas al QTabWidget principal
        
        self.main_tab_widget.addTab(self.history_tab, "Historial")
        self.main_tab_widget.addTab(self.results_tab, "Resultados")
        self.main_tab_widget.addTab(self.comparison_tab, "Comparación")

        # Configurar el contenido de las pestañas
        
        self.setup_history_tab()
        self.setup_results_tab()
        self.setup_comparison_tab()

    def setup_results_tab(self):
        results_layout = QVBoxLayout(self.results_tab)
        results_content = create_main_layout(self)
        results_layout.addWidget(results_content)

    def setup_history_tab(self):

        
        
        # Crear el layout principal
        history_layout = QVBoxLayout(self.history_tab)
        
        # Crear el campo de búsqueda
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Buscar...")
        self.search_box.textChanged.connect(self.filter_table)
        
        # Crear la tabla
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(9)
        self.history_table.setHorizontalHeaderLabels(['Fecha', 'Nombre', 'Identificación', 'Edad', 'Sexo', 'Localización', 'Imagen', 'Clase Predicha', 'Probabilidades'])
        
        # Añadir widgets al layout
        history_layout.addWidget(self.search_box)
        history_layout.addWidget(self.history_table)
        self.populate_table('historial_pacientes.csv') 

    def filter_table(self):
        search_text = self.search_box.text().lower()
        
        for row in range(self.history_table.rowCount()):
            match = False
            for column in range(self.history_table.columnCount()):
                item = self.history_table.item(row, column)
                if item and search_text in item.text().lower():
                    match = True
                    break
            self.history_table.setRowHidden(row, not match)

    def populate_table(self, csv_file):
        # Leer los datos desde el archivo CSV
        df = pd.read_csv(csv_file)
        
        # Configurar el número de filas y columnas de la tabla
        self.history_table.setRowCount(len(df))
        self.history_table.setColumnCount(len(df.columns) + 1)  # +1 para la columna de botones
        
        # Configurar los encabezados de columna
        self.history_table.setHorizontalHeaderLabels(list(df.columns) + ['Descargar'])
        
        for row in range(len(df)):
            for column, item in enumerate(df.iloc[row]):
                table_item = QTableWidgetItem(str(item))
                self.history_table.setItem(row, column, table_item)
            
            # Añadir un botón en la última columna
            button = QPushButton("Descargar")
            button.clicked.connect(lambda checked, r=row: self.on_button_click(r))
            self.history_table.setCellWidget(row, len(df.columns), button)

    @pyqtSlot()
    def on_button_click(self, row):
        # Obtener los datos de la fila seleccionada
        patient_data = {
            'Nombre': self.history_table.item(row, 1).text(),
            'Identificación': self.history_table.item(row, 2).text(),
            'Edad': self.history_table.item(row, 3).text(),
            'Sexo': self.history_table.item(row, 4).text(),
            'Localización': self.history_table.item(row, 5).text()
        }
        result = {
            'clase_predicha': self.history_table.item(row, 7).text(),
            'probabilidades': self.history_table.item(row, 8).text()
        }
        image_path = self.history_table.item(row, 6).text()

        try:
            # Comprobar si el archivo de imagen existe
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"No se puede encontrar la imagen en la ruta: {image_path}")

            # Generar el PDF
            pdf = generate_pdf_report(patient_data, result, image_path)

            # Guardar el PDF
            pdf_filename = f"reporte_{patient_data['Identificación']}.pdf"
            with open(pdf_filename, 'wb') as f:
                f.write(pdf)

            print(f"PDF generado y guardado como {pdf_filename}")

        except FileNotFoundError as e:
            # Mostrar mensaje de error en PyQt6
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            # Mostrar mensaje de error para otras excepciones
            QMessageBox.critical(self, "Error", f"Ha ocurrido un error inesperado: {str(e)}")


    def setup_comparison_tab(self):
        comparison_layout = QVBoxLayout(self.comparison_tab)
        self.comparison_grid = QGridLayout()
        comparison_widget = QWidget()
        comparison_widget.setLayout(self.comparison_grid)
        comparison_layout.addWidget(comparison_widget)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de Imagen (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_viewer.set_image(pixmap)
            self.current_image = file_name
            self.image_history.append(file_name)
            self.result_text.clear()

    def analyze_image(self):
        if not self.current_image:
            self.result_text.setText("<p style='color: red;'>Por favor, cargue una imagen primero.</p>")
            return

        patient_data = {
            "Nombre": self.name_input.text(),
            "Identificación": self.id_input.text(),
            "Edad": self.age_input.text(),
            "Sexo": self.sex_input.currentText(),
            "categoria":"malignant",
            "Localización": self.location_input.currentText()
        }
        result = predicto(self.current_model, self.current_image, patient_data["Edad"], patient_data["Sexo"], patient_data["Localización"], metadata_path='metadatos_T.csv')

        full_class_name = self.cancer_types.get(result['predicted_class'], result['predicted_class'])
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

        for class_name, probability in result['probabilities'].items():
            full_name = self.cancer_types.get(class_name, class_name)
            result_text += f"<li><b>{full_name}:</b> {probability:.4f}</li>"

        result_text += "</ul>"

        max_probability = max(result['probabilities'].values())

        if max_probability > 0.5:
            recommendation = "<p style='color: red;'><b>Se recomienda consultar a un dermatólogo.</b></p>"
        else:
            recommendation = "<p style='color: green;'><b>El riesgo parece bajo, pero consulte a un médico si tiene dudas.</b></p>"

        result_text += recommendation

        self.result_text.setHtml(result_text)

        self.save_to_csv(patient_data, max_probability, full_class_name)

        images_and_descriptions = [
            ('interpretation_DenseNet121.jpg', 'Modelo DenseNet21'),
            ('interpretation_EfficientNetV2B0.jpg', 'Modelo EfficientNet'),
            ('interpretation_InceptionV3.jpg', 'Modelo Inception'),
            ('interpretation_MobileNet.jpg', 'Modelo MobileNet'),
            ('interpretation_ResNet50.jpg', 'Modelo ResNet50'),
            ('interpretation_Xception.jpg', 'Modelo Xeption')
        ]

        self.update_comparison_tab(images_and_descriptions)
        # Guardar la imagen con el nombre del paciente
        if self.current_image:
            
            # Define el directorio base donde se guardarán las imágenes
            base_directory = 'ruta/a/tu/directorio/de/imagenes'
            
            # Obtener el número de identificación del paciente
            patient_id = patient_data["Identificación"]
            
            # Crear un directorio específico para el paciente
            save_directory = os.path.join(base_directory, patient_id)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            
            # Obtener el nombre del archivo original y su extensión
            file_info = QFileInfo(self.current_image)
            base_name = file_info.baseName()
            extension = file_info.suffix()
            
            # Definir el nuevo nombre de archivo basado en el nombre del paciente
            new_file_name = f"{patient_data['Nombre']}.{extension}"
            new_file_path = os.path.join(save_directory, new_file_name)
            
            # Cargar la imagen y guardarla en el nuevo directorio
            pixmap = QPixmap(self.current_image)
            pixmap.save(new_file_path)
        pdf =generate_pdf_report(patient_data, result,new_file_path)
            # Guardar el PDF
        with open('reporte_dermatologico.pdf', 'wb') as f:
            f.write(pdf)

    def update_comparison_tab(self, images_and_descriptions):
            # Limpiar el layout existente
        for i in reversed(range(self.comparison_grid.count())):
            self.comparison_grid.itemAt(i).widget().setParent(None)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for image_file, description in images_and_descriptions:
            # Crear un widget contenedor para cada imagen y descripción
            container = QFrame()
            container.setFrameShape(QFrame.Shape.Box)
            container.setFrameShadow(QFrame.Shadow.Raised)
            container.setLineWidth(2)
            container_layout = QVBoxLayout(container)

            # Crear el visor de gráficos zoomable
            graphics_view = ZoomableGraphicsView()
            scene = QGraphicsScene()
            pixmap = QPixmap(image_file)
            if not pixmap.isNull():
                pixmap_item = scene.addPixmap(pixmap)
                scene.setSceneRect(QRectF(pixmap.rect()))
            else:
                scene.addText("Imagen no encontrada")
            graphics_view.setScene(scene)

            # Configurar el tamaño del visor
            graphics_view.setMinimumSize(300, 300)
            graphics_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # Agregar el visor al contenedor
            container_layout.addWidget(graphics_view)

            # Agregar la descripción
            description_label = QLabel(description)
            description_label.setFont(QFont("Arial", 12))
            description_label.setWordWrap(True)
            description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(description_label)

            # Agregar el contenedor al layout principal
            scroll_layout.addWidget(container)

        scroll_area.setWidget(scroll_widget)
        self.comparison_grid.addWidget(scroll_area, 0, 0, 1, 1)

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

    def update_patient_history_table(self):
        self.history_table.setRowCount(0)

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
        self.update_patient_history_table()

    def update_zoom(self, value):
        if hasattr(self, 'image_viewer'):   
            self.image_viewer.update_zoom(value)

    def update_contrast(self, value):
        if hasattr(self, 'image_viewer'):
            self.image_viewer.update_contrast(value)

    def update_brightness(self, value):
        if hasattr(self, 'image_viewer'):
            self.image_viewer.update_brightness(value)

if __name__ == "__main__":
    app = QApplication([])
    window = MelanomaDetector()
    window.show()
    app.exec()
