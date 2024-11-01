import os
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QHBoxLayout, QLineEdit, QWidget, QPushButton,QVBoxLayout, QFileDialog, QLabel, QTableWidgetItem, QTableWidget, QApplication, QGridLayout, QTabWidget, QGraphicsScene
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, pyqtSlot, QFile,QTimer
from ui_components import create_main_layout
from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply, QHttpMultiPart, QHttpPart
from PyQt6.QtCore import QUrl, QIODevice
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
import json
from PyQt6.QtCore import QFileInfo
from PyQt6.QtWidgets import  QGraphicsView, QGridLayout
from PyQt6.QtGui import QPixmap, QFont, QWheelEvent
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QMessageBox
import sys
from login import LoginWindow
from utils import get_patient_directory, get_interpretation_directory
import os


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




class NetworkError(Exception):
    """Excepción personalizada para errores de red"""
    pass

class FileError(Exception):
    """Excepción personalizada para errores de archivo"""
    pass

class PDFGenerationError(Exception):
    """Excepción personalizada para errores en la generación del PDF"""
    pass



class MelanomaDetector(QMainWindow):
    def __init__(self, user_type,show_login_callback):
        super().__init__()
        self.user_type = user_type
        self.show_login_callback = show_login_callback
        self.base_url = "http://54.227.28.76:8000/" 

        self.network_manager = QNetworkAccessManager()
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

        # Crear un layout horizontal para la parte superior
        top_layout = QHBoxLayout()

        # Crear el botón de cierre de sesión
        self.logout_button = QPushButton("Cerrar sesión")
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.setStyleSheet("background-color: #ff9999; padding: 5px;")  # Estilo para hacerlo más visible

        # Añadir el botón al layout superior, alineado a la derecha
        top_layout.addStretch(1)  # Añadir un stretch para empujar el botón a la derecha
        top_layout.addWidget(self.logout_button)

        # Añadir el layout superior al layout principal
        main_layout.addLayout(top_layout)

        # Crear el QTabWidget principal
        self.main_tab_widget = QTabWidget()
        main_layout.addWidget(self.main_tab_widget)

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
        url = QUrl(f"{self.base_url}/get_predictions")
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
            
            # Result 
            self.history_table.setItem(row_position, 9, QTableWidgetItem(prediction.get('predicted_class', '')))
            self.history_table.setItem(row_position, 10, QTableWidgetItem(str(prediction['diagnostico']['observacion'])))
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
        
        try:
            patient_identification = self.history_table.item(row, 2).text()  # Número de cédula del paciente
            
            # Crear request para listar PDFs
            url = QUrl(f"{self.base_url}/pdfs/")
            request = QNetworkRequest(url)
            
            # Realizar la petición GET
            reply = self.network_manager.get(request)
            reply.finished.connect(lambda: self.handle_pdf_list_response(reply, patient_identification))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ha ocurrido un error inesperado: {str(e)}")
        
    def handle_pdf_list_response(self, reply: QNetworkReply, patient_id: str):
        """Maneja la respuesta de la lista de PDFs disponibles"""
        try:
            if reply.error() == QNetworkReply.NetworkError.NoError:
                # Leer y parsear la respuesta JSON
                data = json.loads(str(reply.readAll(), 'utf-8'))
                
                # Filtrar PDFs por ID de paciente
                patient_pdfs = [pdf for pdf in data if pdf['patient_id'] == patient_id]
                
                if not patient_pdfs:
                    QMessageBox.warning(
                        self,  # Cambié self.parent a self
                        "No se encontraron PDFs",
                        f"No se encontraron PDFs para el paciente con ID: {patient_id}"
                    )
                    return
                
                # Usar el PDF más reciente (basado en el timestamp)
                latest_pdf = sorted(patient_pdfs, key=lambda x: x['timestamp'], reverse=True)[0]
                
                # Solicitar ubicación de guardado
                save_path, _ = QFileDialog.getSaveFileName(
                    self,  # Cambié self.parent a self
                    "Guardar PDF",
                    f"reporte_dermatologico_{latest_pdf['filename']}",
                    "PDF Files (*.pdf);;All Files (*)"
                )
                
                if save_path:
                    # Iniciar la descarga del PDF
                    pdf_url = QUrl(f"{self.base_url}/pdfs/{latest_pdf['patient_id']}/{latest_pdf['timestamp']}/{latest_pdf['filename']}")
                    pdf_request = QNetworkRequest(pdf_url)
                    pdf_reply = self.network_manager.get(pdf_request)
                    
                    # Conectar la finalización de la descarga con el guardado del archivo
                    pdf_reply.finished.connect(
                        lambda: self.handle_pdf_download(pdf_reply, save_path)
                    )
            else:
                QMessageBox.critical(
                    self,  # Cambié self.parent a self
                    "Error",
                    f"Error al obtener la lista de PDFs: {reply.errorString()}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,  # Cambié self.parent a self
                "Error",
                f"Error procesando la respuesta: {str(e)}"
            )
        finally:
            reply.deleteLater()

    def handle_pdf_download(self, reply: QNetworkReply, save_path: str):
        """Maneja la descarga y guardado del PDF"""
        try:
            if reply.error() == QNetworkReply.NetworkError.NoError:
                # Leer los datos del PDF
                pdf_data = reply.readAll()
                
                # Guardar el PDF
                with open(save_path, 'wb') as f:
                    f.write(pdf_data.data())
                
                QMessageBox.information(
                    self,  # Cambié self.parent a self
                    "PDF Descargado",
                    f"PDF guardado exitosamente en:\n{save_path}"
                )
            else:
                QMessageBox.critical(
                    self,  # Cambié self.parent a self
                    "Error",
                    f"Error al descargar el PDF: {reply.errorString()}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,  # Cambié self.parent a self
                "Error",
                f"Error al guardar el PDF: {str(e)}"
            )
        finally:
            reply.deleteLater()


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

        url = QUrl(f"{self.base_url}/predict")
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
            try:
                # Intenta decodificar como UTF-8
                response_data = reply.readAll().data()
                print(f"Datos de la respuesta: {response_data[:100]}")  # Verifica el contenido de la respuesta
                result = json.loads(response_data.decode('utf-8'))
            except UnicodeDecodeError as e:
                print(f"Error decodificando la respuesta: {str(e)}")
                QMessageBox.critical(self, "Error de Decodificación", "La respuesta no está en formato UTF-8")
                return
            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON: {str(e)}")
                QMessageBox.critical(self, "Error de Formato", "La respuesta no es un JSON válido")
                return

            # Añadir campo observación
            result['observacion'] = self.current_patient_data.get('observacion', 'No se proporcionó observación.')
            
            # Obtener el ID del paciente y la fecha de diagnóstico
            patient_id = self.current_patient_data.get('identification', None)
            if not patient_id:
                QMessageBox.critical(self, "Error", "No se encontró el ID del paciente.")
                return
            
            diagnosis_date = datetime.now()

            try:
                # Generar directorios y añadirlos a result
                patient_dir = get_patient_directory(patient_id, diagnosis_date)
                interpretation_dir = get_interpretation_directory(patient_id, diagnosis_date)
                
                # Verificar si los directorios son válidos
                if not patient_dir or not interpretation_dir:
                    raise ValueError("No se pudieron generar los directorios.")

                # Actualizar el resultado con los nuevos campos
                result.update({
                    'patient_dir': patient_dir,
                    'interpretation_dir': interpretation_dir,
                    'diagnosis_date': diagnosis_date.strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"Error generando los directorios: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error generando directorios: {str(e)}")
                return
            
            # Verificar que el resultado contiene los campos esperados
            if 'patient_dir' not in result or 'interpretation_dir' not in result:
                print(f"Falta el campo requerido en el resultado: {result}")
                QMessageBox.critical(self, "Error", "Faltan campos en los resultados para generar el PDF.")
                return

            # Mostrar el resultado para depuración
            print(f"Resultado antes de la generación del PDF: {result}")

            # Llamar a la función de generación de PDFs
            self.generate_and_save_pdf(result)
            self.display_result(result, self.current_patient_data)
            self.load_initial_data()
        else:
            self.show_network_error(reply)



    def generate_and_save_pdf(self, input_result):
        try:
            # Reestructurar los datos en el formato esperado
            storage_info = input_result.get("storage_info", {})
            formatted_result = {
                "result": {
                    "paciente": input_result["paciente"],
                    "diagnostico": input_result["diagnostico"],
                    "imagen": input_result["imagen"],
                    "predicted_class": input_result["predicted_class"],
                    "probabilities": input_result["probabilities"]
                },
                "patient_dir": storage_info.get("patient_dir", ""),
                "diagnosis_date": storage_info.get("diagnosis_date", ""),
                "original_image_path": input_result["imagen"]["ruta_imagen"]
            }

            try:
                print("Configurando la solicitud al servidor para generar el PDF...")
                url = QUrl(f"{self.base_url}/generate-pdf")
            
                
                request = QNetworkRequest(url)
                request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
                
                # Asegúrate de que network_manager sea un atributo de la clase
                if not hasattr(self, 'network_manager'):
                    self.network_manager = QNetworkAccessManager(self)

                # Conectar la respuesta a handle_pdf_response
                self.network_manager.finished.disconnect()  # Desconectar cualquier conexión anterior
                self.network_manager.finished.connect(self.handle_pdf_response)

                # Prepara y valida el JSON
                self.current_pdf_result = formatted_result
                json_data = json.dumps(formatted_result, indent=4)  # Formato legible
                print("Datos enviados al servidor (JSON):")
                print(json_data)

                # Enviar la solicitud
                self.current_reply = self.network_manager.post(request, json_data.encode('utf-8'))
                print("Solicitud de generación de PDF enviada al servidor...")

                # Configurar timeout para la solicitud
                self.timeout_timer = QTimer(self)
                self.timeout_timer.setSingleShot(True)
                self.timeout_timer.timeout.connect(lambda: self.handle_request_timeout(self.current_reply))
                self.timeout_timer.start(30000)  # 30 segundos de timeout

            except Exception as e:
                raise NetworkError(f"Error enviando la solicitud al servidor: {str(e)}")

        except ValueError as e:
            QMessageBox.warning(self, "Error de Validación", str(e))
        except FileError as e:
            QMessageBox.critical(self, "Error de Archivo", str(e))
        except NetworkError as e:
            QMessageBox.critical(self, "Error de Red", str(e))
        except PDFGenerationError as e:
            QMessageBox.critical(self, "Error Generando PDF", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error Inesperado", f"Ocurrió un error inesperado: {str(e)}")


    def handle_request_timeout(self, reply):
        if reply and reply.isRunning():
            reply.abort()
            self.clean_up_timeout()
            QMessageBox.critical(self, "Error de Timeout", "La solicitud al servidor ha excedido el tiempo de espera")


    def handle_pdf_response(self, reply):
        try:
            print("Procesando respuesta del servidor...")

            if reply.error() == QNetworkReply.NetworkError.NoError:
                # Verificar el tipo de contenido
                content_type = reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader)
                if content_type != "application/pdf":
                    raise PDFGenerationError(f"El servidor no devolvió un PDF (tipo de contenido: {content_type})")

                # Obtener el contenido del PDF
                pdf_data = reply.readAll()
                if not pdf_data:
                    raise PDFGenerationError("El servidor devolvió un PDF vacío")

                # Guardar el PDF
                try:
                    pdf_path = os.path.join(self.current_pdf_result['patient_dir'], "reporte_dermatologico.pdf")
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_data)
                    
                    QMessageBox.information(
                        self, 
                        "PDF Generado", 
                        f"Se ha generado el reporte PDF:\n{pdf_path}")

                except Exception as e:
                    raise FileError(f"Error guardando el PDF: {str(e)}")

            else:
                # Manejar diferentes tipos de errores de red
                error_code = reply.error()
                error_msg = reply.errorString()
                print(f"Error de red: {error_msg} (código: {error_code})")
                raise NetworkError(f"Error de red: {error_msg} (código: {error_code})")

        except NetworkError as e:
            QMessageBox.critical(self, "Error de Red", str(e))
        except FileError as e:
            QMessageBox.critical(self, "Error de Archivo", str(e))
        except PDFGenerationError as e:
            QMessageBox.critical(self, "Error Generando PDF", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error Inesperado", f"Error procesando la respuesta del servidor: {str(e)}")
        finally:
            # Limpiar recursos
            if hasattr(self, 'timeout_timer'):
                self.timeout_timer.stop()
                self.timeout_timer.deleteLater()
            if reply:
                reply.deleteLater()
            if hasattr(self, 'current_reply'):
                delattr(self, 'current_reply')
            if hasattr(self, 'current_pdf_result'):
                delattr(self, 'current_pdf_result')

    # Funciones auxiliares

    def validate_result_fields(self, result, required_fields):
        if not result:
            raise ValueError("No se proporcionaron resultados para generar el PDF")
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Falta el campo requerido: {field}")


    def create_patient_directory(self, patient_dir):
        try:
            os.makedirs(patient_dir, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise FileError(f"Error creando el directorio {patient_dir}: {str(e)}")


    def process_original_image(self, patient_dir):
        if not self.current_image or not os.path.exists(self.current_image):
            raise FileError("No se ha cargado o encontrado ninguna imagen para el análisis")
        
        file_info = QFileInfo(self.current_image)
        extension = file_info.suffix()
        if not extension:
            raise FileError("La imagen original no tiene una extensión válida")
        
        new_file_name = f"original_image.{extension}"
        new_file_path = os.path.join(patient_dir, new_file_name)
        
        if QFile.exists(new_file_path):
            QFile.remove(new_file_path)
        
        if not QFile.copy(self.current_image, new_file_path):
            raise FileError(f"No se pudo copiar la imagen original a {new_file_path}")
        
        return new_file_path


    def prepare_request_data(self, pdf_request_data):
        try:
            # Verificar codificación en UTF-8
            for key, value in pdf_request_data.items():
                if isinstance(value, str):
                    value.encode('utf-8').decode('utf-8')
            return json.dumps(pdf_request_data)
        except Exception as e:
            raise PDFGenerationError(f"Error preparando los datos para el PDF: {str(e)}")


    def send_pdf_request(self, json_data):
        url = QUrl(f"{self.base_url}/generate-pdf")

        
        request = QNetworkRequest(url)
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")

        if not hasattr(self, 'network_manager'):
            self.network_manager = QNetworkAccessManager(self)
            self.network_manager.finished.connect(self.handle_pdf_response)

        self.current_reply = self.network_manager.post(request, json_data.encode())
        self.setup_timeout(self.current_reply)


    def setup_timeout(self, reply):
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(lambda: self.handle_request_timeout(reply))
        self.timeout_timer.start(30000)  # 30 segundos


    def clean_up_reply(self, reply):
        if hasattr(self, 'timeout_timer'):
            self.timeout_timer.stop()
            self.timeout_timer.deleteLater()
        if reply:
            reply.deleteLater()
        if hasattr(self, 'current_reply'):
            delattr(self, 'current_reply')
        if hasattr(self, 'current_pdf_result'):
            delattr(self, 'current_pdf_result')


    def clean_up_timeout(self):
        if hasattr(self, 'timeout_timer'):
            self.timeout_timer.stop()
            self.timeout_timer.deleteLater()


    def process_pdf_response(self, reply):
        content_type = reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader)
        if content_type != "application/pdf":
            raise PDFGenerationError(f"El servidor no devolvió un PDF (tipo de contenido: {content_type})")
        
        pdf_data = reply.readAll()
        if not pdf_data:
            raise PDFGenerationError("El servidor devolvió un PDF vacío")
        
        pdf_path = os.path.join(self.current_pdf_result['patient_dir'], "reporte_dermatologico.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        QMessageBox.information(self, "PDF Generado", f"Se ha generado el reporte PDF:\n{pdf_path}")


    def show_network_error(self, reply):
        error_code = reply.error()
        error_msg = reply.errorString()
        raise NetworkError(f"Error de red: {error_msg} (código: {error_code})")

    def save_to_database(self):
       
        print("Datos guardados en la base de datos con éxito")
        self.update_patient_history_table()


    def update_patient_history_table(self):
        self.history_network_manager = QNetworkAccessManager()
        self.history_network_manager.finished.connect(self.handle_history_response)

        url = QUrl(f"{self.base_url}/get_predictions")
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
        # Crear el network manager si no existe
        self.comparison_network_manager = QNetworkAccessManager()
        
        # Inicializar el diccionario para rastrear las solicitudes y sus posiciones
        self.pending_image_positions = {}
        self.request_to_index = {}  # Nuevo diccionario para mapear solicitudes a índices
        self.total_images = 0
        
        # Conectar el manejador específico para la lista de imágenes
        self.comparison_network_manager.finished.connect(self.handle_image_list_response)

        # Preparar la solicitud para obtener los nombres de las imágenes
        url = QUrl(f"{self.base_url}/imagenes/")
        request = QNetworkRequest(url)

        # Realizar la solicitud GET
        self.comparison_network_manager.get(request)

    def handle_image_list_response(self, reply):
        # Desconectar el manejador de lista
        self.comparison_network_manager.finished.disconnect(self.handle_image_list_response)
        
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                json_data = reply.readAll().data().decode('utf-8')
                data = json.loads(json_data)
                images = data.get("imagenes", [])
                self.total_images = len(images)
                print(f"Total de imágenes a cargar: {self.total_images}")

                # Limpiar el layout existente
                for i in reversed(range(self.comparison_grid.count())): 
                    self.comparison_grid.itemAt(i).widget().setParent(None)

                # Conectar el manejador para las imágenes individuales
                self.comparison_network_manager.finished.connect(self.handle_image_response)

                # Descargar cada imagen individualmente
                for index, image_name in enumerate(images):
                    image_url = QUrl(f"{self.base_url}/imagenes/{image_name}")
                    request = QNetworkRequest(image_url)
                    
                    model_name = image_name.split('_')[0].capitalize()
                    description_label = QLabel(f"Modelo: {model_name}")
                    description_label.setFont(QFont("Arial", 12))
                    description_label.setWordWrap(True)
                    
                    row = index // 2
                    col = index % 2
                    self.comparison_grid.addWidget(description_label, row * 2 + 1, col, 1, 1, Qt.AlignmentFlag.AlignCenter)
                    
                    self.pending_image_positions[index] = (row * 2, col)
                    self.request_to_index[image_url.toString()] = index
                    
                    print(f"Solicitando imagen {index}: {image_url.toString()}")
                    self.comparison_network_manager.get(request)

            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error al procesar la lista de imágenes: {str(e)}")
            except Exception as e:
                print(f"Error procesando la respuesta: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error al procesar las imágenes: {str(e)}")
        else:
            print(f"Error en la solicitud de red: {reply.errorString()}")
            QMessageBox.critical(self, "Error", f"Error al obtener las imágenes: {reply.errorString()}")

    def handle_image_response(self, reply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                url = reply.url().toString()
                index = self.request_to_index.get(url)
                
                if index is not None and index in self.pending_image_positions:
                    row, col = self.pending_image_positions[index]
                    
                    image_data = reply.readAll()
                    pixmap = QPixmap()
                    
                    if pixmap.loadFromData(image_data):
                        scene = QGraphicsScene() # type: ignore
                        scene.addPixmap(pixmap)
                        
                        zoomable_view = ZoomableGraphicsView()
                        zoomable_view.setScene(scene)
                        zoomable_view.setFixedSize(300, 300)
                        
                        self.comparison_grid.addWidget(zoomable_view, row, col, 1, 1, Qt.AlignmentFlag.AlignCenter)
                        
                        del self.pending_image_positions[index]
                        del self.request_to_index[url]
                        
                        print(f"Imagen {index} cargada correctamente. Quedan {len(self.pending_image_positions)} imágenes por cargar.")
                        
                        if not self.pending_image_positions:
                            print("Todas las imágenes han sido cargadas.")
                            self.comparison_network_manager.finished.disconnect(self.handle_image_response)
                    else:
                        print(f"Error: La imagen {index} no se pudo cargar.")
                else:
                    print(f"Error: No se encontró la posición para la URL {url}")

            except Exception as e:
                print(f"Error procesando la imagen: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error al procesar la imagen: {str(e)}")
        else:
            print(f"Error en la solicitud de imagen: {reply.errorString()}")
            QMessageBox.critical(self, "Error", f"Error al descargar la imagen: {reply.errorString()}")

    def cleanup_comparison_tab(self):
        """Limpia los recursos cuando se cierra la pestaña"""
        if hasattr(self, 'comparison_network_manager'):
            self.comparison_network_manager.finished.disconnect()
            self.pending_image_positions.clear()
            self.request_to_index.clear()

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
