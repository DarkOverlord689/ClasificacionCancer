import os
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QLineEdit, QWidget, QPushButton,QVBoxLayout, QFileDialog, QLabel, QTableWidgetItem, QTableWidget, QApplication, QGridLayout, QTabWidget
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, pyqtSlot, QFile
from ui_components import create_main_layout
from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply, QHttpMultiPart, QHttpPart
from PyQt6.QtCore import QUrl, QByteArray, QBuffer, QIODevice
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PyQt6.QtCore import QUrlQuery
import json
from PyQt6.QtCore import QFileInfo
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QFrame, QWidget, QGridLayout
from PyQt6.QtGui import QPixmap, QFont, QWheelEvent
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QScrollArea, QSizePolicy
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QMessageBox
import sys
from pdfgenerator import generate_pdf_report
from login import LoginWindow
import ast
from utils import get_patient_directory, get_interpretation_directory
import os
import shutil
import re

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




class CustomTabWidget(QTabWidget):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Tab:
            # Cambia de pestaña al presionar Tab
            current_index = self.currentIndex()
            next_index = (current_index + 1) % self.count()
            self.setCurrentIndex(next_index)
            event.accept()  # Aceptar el evento para evitar el comportamiento predeterminado
        else:
            super().keyPressEvent(event)  # Llamar al comportamiento predeterminado

class MelanomaDetector(QMainWindow):
    def __init__(self, user_type,show_login_callback):
        super().__init__()
        self.user_type = user_type
        self.show_login_callback = show_login_callback
        self.models = {
            'DenseNet': 'best_model_DenseNet121.h5',
            'ResNet': 'best_model_ResNet50.h5',
            'Xception': 'best_model_Xception.h5',
            'MobileNet': 'best_model_MobileNet.h5',
            'Inception': 'best_model_InceptionV3.h5',
            'EfficientNet': 'best_model_EfficientNetV2B0.h5'
        }

        self.images_and_descriptions = [
            ('interpretation_DenseNet121.jpg', 'Modelo DenseNet21'),
            ('interpretation_EfficientNetV2B0.jpg', 'Modelo EfficientNet'),
            ('interpretation_InceptionV3.jpg', 'Modelo Inception'),
            ('interpretation_MobileNet.jpg', 'Modelo MobileNet'),
            ('interpretation_ResNet50.jpg', 'Modelo ResNet50'),
            ('interpretation_Xception.jpg', 'Modelo Xeption')
        ]

        self.current_model = None
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

        # Crear el botón de cierre de sesión
        self.logout_button = QPushButton("Cerrar sesión")
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.setStyleSheet("background-color: #ff9999; padding: 5px;")  # Estilo para hacerlo más visible

        # Añadir el botón de cierre de sesión al principio del layout
        main_layout.addWidget(self.logout_button)

        # Crear las pestañas principales
        self.history_tab = QWidget()
        self.results_tab = QWidget()
        self.comparison_tab = QWidget()

        # Añadir las pestañas al QTabWidget principal
        self.main_tab_widget.addTab(self.history_tab, "Historial")
        
        if self.user_type == 'admin':
            self.main_tab_widget.addTab(self.results_tab, "Registro")
            self.main_tab_widget.addTab(self.comparison_tab, "Comparación")

        
        self.setup_history_tab()
        if self.user_type == 'admin':
            self.setup_results_tab()
            
            self.setup_comparison_tab()

    def logout(self):
        self.close()
        self.show_login_callback()


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
        self.history_table.setColumnCount(12)  
        self.history_table.setHorizontalHeaderLabels([
            'Fecha', 'Nombre', 'Identificación', 'Edad', 'Sexo', 
            'Localización', 'Tipo de Cáncer', 'Imagen', 'Probabilidades', 'Clase Predicha', 'Observaciones', 'Descargar'
        ])
        
        # Añadir widgets al layout
        history_layout.addWidget(self.search_box)
        history_layout.addWidget(self.history_table)
        
        # Cargar los datos iniciales
        self.load_initial_data()

    


    def load_initial_data(self):
        # Crear un QNetworkAccessManager para la solicitud
        self.initial_data_manager = QNetworkAccessManager()
        self.initial_data_manager.finished.connect(self.handle_initial_data_response)

        # Preparar la solicitud
        url = QUrl("http://localhost:8000/get_predictions")
        request = QNetworkRequest(url)

        # Enviar la solicitud
        self.initial_data_manager.get(request)

    def handle_initial_data_response(self, reply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = json.loads(str(reply.readAll(), 'utf-8'))
            self.populate_history_table(data)
        else:
            print(f"Error al obtener los datos iniciales: {reply.errorString()}")
            QMessageBox.warning(self, "Error", "No se pudieron cargar los datos iniciales. Por favor, intente de nuevo más tarde.")

    def populate_history_table(self, data):
        self.history_table.setRowCount(0)
        for prediction in data:
            row_position = self.history_table.rowCount()
            self.history_table.insertRow(row_position)
            
            # Asumiendo que 'paciente' contiene la información del paciente
            self.history_table.setItem(row_position, 0, QTableWidgetItem(prediction['paciente']['fecha_registro']))
            self.history_table.setItem(row_position, 1, QTableWidgetItem(prediction['paciente']['nombre']))
            self.history_table.setItem(row_position, 2, QTableWidgetItem(str(prediction['paciente']['numero_identificacion'])))
            self.history_table.setItem(row_position, 3, QTableWidgetItem(str(prediction['paciente']['edad'])))
            self.history_table.setItem(row_position, 4, QTableWidgetItem(prediction['paciente']['sexo']))
            
            # Asumiendo que 'diagnostico' contiene la información del diagnóstico
            self.history_table.setItem(row_position, 5, QTableWidgetItem(prediction['diagnostico']['localizacion']))
            self.history_table.setItem(row_position, 6, QTableWidgetItem(prediction['diagnostico']['tipo_cancer']))
            
            if prediction.get('imagen') is not None:
                self.history_table.setItem(row_position, 7, QTableWidgetItem(prediction['imagen']['ruta_imagen']))
            else:
                self.history_table.setItem(row_position, 7, QTableWidgetItem("No image"))

            # Probabilidades
            probabilities = prediction.get('probabilities', {})
            self.history_table.setItem(row_position, 8, QTableWidgetItem(str(probabilities)))
            
            # Result (asumiendo que es el mismo que 'predicted_class')
            self.history_table.setItem(row_position, 9, QTableWidgetItem(prediction.get('predicted_class', '')))
            self.history_table.setItem(row_position, 10, QTableWidgetItem(prediction.get('predicted_class', '')))
            # Añadir botón de descarga
            download_button = QPushButton("Descargar PDF")
            download_button.clicked.connect(lambda _, row=row_position: self.on_button_click(row))
            self.history_table.setCellWidget(row_position, 11, download_button)

    def filter_table(self):
        search_text = self.search_box.text().lower()
        
        for row in range(self.history_table.rowCount()):
            item = self.history_table.item(row, 2)  # Columna del número de identificación
            if item and search_text in item.text().lower():
                self.history_table.setRowHidden(row, False)
            else:
                self.history_table.setRowHidden(row, True)

    @pyqtSlot()
    def on_button_click(self, row): 

        """ Esta función se encarga de descargar los pdf cuando se pulsa el botón en la tabla
         de historial """

        patient_identification = self.history_table.item(row, 2).text()  # Número de cédula del paciente
        pdf_folder = os.path.join('datos_paciente', patient_identification)  # Ruta a la carpeta del paciente

        try:
            # Listar subcarpetas en pdf_folder
            subfolders = [f for f in os.listdir(pdf_folder) if os.path.isdir(os.path.join(pdf_folder, f))]

            if not subfolders:
                raise FileNotFoundError(f"No se encontraron subcarpetas en: {pdf_folder}")

            # Usar la primera subcarpeta encontrada
            subfolder = os.path.join(pdf_folder, subfolders[0])

            # Buscar el archivo PDF con el formato especificado
            pdf_pattern = re.compile(rf"reporte_dermatologico_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}\.pdf")
            pdf_file = None

            for file in os.listdir(subfolder):
                if pdf_pattern.match(file):
                    pdf_file = file
                    break

            if pdf_file is None:
                raise FileNotFoundError(f"No se encontró un archivo PDF con el formato esperado en: {subfolder}")

            pdf_path = os.path.join(subfolder, pdf_file)  # Ruta completa del PDF a buscar

            # Abrir un cuadro de diálogo para elegir la ubicación de guardado
            save_path, _ = QFileDialog.getSaveFileName(self, "Guardar PDF", "", "PDF Files (*.pdf);;All Files (*)")

            if save_path:
                # Copiar el archivo PDF a la nueva ubicación
                shutil.copy(pdf_path, save_path)
                QMessageBox.information(self, "PDF Generadoiado", f"PDF creado y guardado como {save_path}")

        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
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
            self.current_image = file_name
            pixmap = QPixmap(file_name)
            self.image_viewer.set_image(pixmap)
            self.image_viewer.zoom_factor = 1
            self.image_viewer.update_image()
            self.image_history.append(file_name)
            self.result_text.clear()

    def analyze_image(self):
        if not self.current_image:
            self.result_text.setText("<p style='color: red;'>Por favor, cargue una imagen primero.</p>")
            return

        patient_data = self.get_patient_data()
        self.send_analysis_request(patient_data)

    def get_patient_data(self):
        return {
            "name": self.name_input.text(),
            "identification": self.id_input.text(),
            "age": self.age_input.text(),
            "sex": self.sex_input.currentText(),
            "localization": self.location_input.currentText(),
            "observacion": self.observation_input.toPlainText()
        }

    def send_analysis_request(self, patient_data):
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.handle_analysis_response)

        url = QUrl("http://localhost:8000/predict")
        request = QNetworkRequest(url)

        multipart = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)

        for key, value in patient_data.items():
            text_part = QHttpPart()
            text_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, f'form-data; name="{key}"')
            text_part.setBody(str(value).encode())  
            multipart.append(text_part)

        # Añadir la imagen
        image_part = QHttpPart()
        image_part.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/jpeg")
        image_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="file"; filename="image.jpg"')
        image = QFile(self.current_image)
        if not image.open(QIODevice.OpenModeFlag.ReadOnly):
            QMessageBox.critical(self, "Error", f"No se pudo abrir la imagen: {self.current_image}")
            return
        
        image_part.setBodyDevice(image)
        image.setParent(multipart)
        multipart.append(image_part)

        reply = self.network_manager.post(request, multipart)
        multipart.setParent(reply)

        # Guardar patient_data para usarlo en handle_analysis_response
        self.current_patient_data = patient_data
            
       

    def handle_analysis_response(self, reply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            result = json.loads(str(reply.readAll(), 'utf-8'))
            
            # Añadir el campo "observación" desde el formulario
            result['observacion'] = self.current_patient_data.get('observacion', 'No se proporcionó observación.')
            # Generar directorios y añadir información al resultado
            patient_id = self.current_patient_data['identification']
            diagnosis_date = datetime.now()
            patient_dir = get_patient_directory(patient_id, diagnosis_date)
            interpretation_dir = get_interpretation_directory(patient_id, diagnosis_date)
            
            result['patient_dir'] = patient_dir
            result['interpretation_dir'] = interpretation_dir
            result['diagnosis_date'] = diagnosis_date.strftime("%Y-%m-%d %H:%M:%S")
            
            self.display_result(result, self.current_patient_data)
            self.generate_and_save_pdf(result)
            self.load_initial_data()
        else:
            error_msg = reply.errorString()
            print(f"Error en la solicitud: {error_msg}")
            QMessageBox.critical(self, "Error", f"Error en la red: {error_msg}")


    def generate_and_save_pdf(self, result):
        patient_dir = result['patient_dir']
        interpretation_dir = result['interpretation_dir']

        os.makedirs(patient_dir, exist_ok=True)

        if self.current_image:
            # Copiar la imagen original del paciente
            file_info = QFileInfo(self.current_image)
            extension = file_info.suffix()
            new_file_name = f"original_image.{extension}"
            new_file_path = os.path.join(patient_dir, new_file_name)
            QFile.copy(self.current_image, new_file_path)

            # Generar el PDF
            try:
                # Reúne las imágenes de interpretación
                interpretation_images = [os.path.join(interpretation_dir, filename) for filename in os.listdir(interpretation_dir) if filename.endswith('.jpg')]
                
                # Llamar a la función para generar el PDF, pasando las imágenes de interpretación
                pdf = generate_pdf_report(result, new_file_path, patient_dir, interpretation_images)

                # Guardar el PDF
                diagnosis_date_str = result['diagnosis_date'].replace(':', '-').replace(' ', '_')
                pdf_file_name = f"reporte_dermatologico_{diagnosis_date_str}.pdf"
                pdf_file_path = os.path.join(patient_dir, pdf_file_name)

                with open(pdf_file_path, 'wb') as f:
                    f.write(pdf)

                QMessageBox.information(self, "PDF Generado", f"Se ha generado el reporte PDF: {pdf_file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo generar el PDF: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "No se ha cargado ninguna imagen para el análisis.")




    def save_to_database(self):
       
        print("Datos guardados en la base de datos con éxito")
        self.update_patient_history_table()


    def update_patient_history_table(self):
        self.history_network_manager = QNetworkAccessManager()
        self.history_network_manager.finished.connect(self.handle_history_response)

        url = QUrl("http://localhost:8000/get_predictions")
        request = QNetworkRequest(url)

        self.history_network_manager.get(request)


    def handle_history_response(self, reply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = json.loads(str(reply.readAll(), 'utf-8'))
            self.populate_history_table(data)
        else:
            print(f"Error al obtener el historial: {reply.errorString()}")


    

    def display_result(self, result, patient_data):
        full_class_name = self.cancer_types.get(result['predicted_class'], result['predicted_class'])
        result_text = self.format_result_text(result, patient_data, full_class_name)
        self.result_text.setHtml(result_text)

        max_probability = max(result['probabilities'].values())
        #self.save_to_csv(patient_data, max_probability, full_class_name)

        images_and_descriptions = [
            ('interpretation_DenseNet121.jpg', 'Modelo DenseNet21'),
            ('interpretation_EfficientNetV2B0.jpg', 'Modelo EfficientNet'),
            ('interpretation_InceptionV3.jpg', 'Modelo Inception'),
            ('interpretation_MobileNet.jpg', 'Modelo MobileNet'),
            ('interpretation_ResNet50.jpg', 'Modelo ResNet50'),
            ('interpretation_Xception.jpg', 'Modelo Xeption')
        ]
        self.update_comparison_tab(images_and_descriptions)

    def format_result_text(self, result, patient_data, full_class_name):
        result_text = f"""
        <h3>Resultados del Análisis</h3>
        <p><b>Nombre:</b> {patient_data['name']} <br>
        <b>Identificación:</b> {patient_data['identification']} <br>
        <b>Edad:</b> {patient_data['age']} <br>
        <b>Sexo:</b> {patient_data['sex']} <br>
        <b>Localización:</b> {patient_data['localization']} <br>
        <b>Observación:</b> {patient_data['observacion']}</p>
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
        return result_text

    def update_comparison_tab(self, images_and_descriptions):
        # Limpiar el layout existente
        for i in reversed(range(self.comparison_grid.count())): 
            self.comparison_grid.itemAt(i).widget().setParent(None)

        for index, (image_file, description) in enumerate(images_and_descriptions):
            row = index // 2
            col = index % 2
            
            image_label = QLabel()
            pixmap = QPixmap(image_file)
            if not pixmap.isNull():
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                image_label.setText("Imagen no encontrada")
            self.comparison_grid.addWidget(image_label, row * 2, col, 1, 1, Qt.AlignmentFlag.AlignCenter)
            
            description_label = QLabel(description)
            description_label.setFont(QFont("Arial", 12))
            description_label.setWordWrap(True)
            self.comparison_grid.addWidget(description_label, row * 2 + 1, col, 1, 1, Qt.AlignmentFlag.AlignCenter)

    

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
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec())
