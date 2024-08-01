from PyQt6.QtWidgets import (QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QSlider, QTabWidget, QTextEdit, QListWidget, QTableView, QWidget,
                             QLineEdit, QFormLayout, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from image_viewer import ImageViewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

def create_left_layout(parent):
    layout = QVBoxLayout()



    # Logo y título
    title_label = QLabel('Clasificador multiclase de Cáncer de piel')
    title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
    layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
    
    logo_label = QLabel()
    logo_pixmap = QPixmap('logo.png')  # Asegúrate de que 'logo.png' exista en el directorio
    if not logo_pixmap.isNull():
        logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
    else:
        logo_label.setText("Logo not found")
    layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

  

    # Formulario de datos del paciente
    form_layout = QFormLayout()
    
    parent.name_input = QLineEdit()
    form_layout.addRow("Nombre:", parent.name_input)
    
    parent.id_input = QLineEdit()
    form_layout.addRow("Identificación:", parent.id_input)
    
    parent.age_input = QLineEdit()
    form_layout.addRow("Edad:", parent.age_input)
    
    parent.sex_input = QComboBox()
    parent.sex_input.addItems(["Masculino", "Femenino", "Otro"])
    form_layout.addRow("Sexo:", parent.sex_input)
    
    parent.location_input = QComboBox()
    parent.location_input.addItems(["Cabeza", "Cuello", "Tronco", "Brazos", "Piernas", "Otro"])
    form_layout.addRow("Localización:", parent.location_input)
    
    layout.addLayout(form_layout)

    # Resto de los componentes
    
    """parent.model_selector = QComboBox()
    layout.addWidget(QLabel("Seleccionar Modelo:"))
    layout.addWidget(parent.model_selector)"""

      # Create a horizontal layout for the label and combobox
    preprocess_layout = QHBoxLayout()

    # Add the label and combobox to the horizontal layout
    preprocess_layout.addWidget(QLabel("Preprocesamiento:"))
    parent.preprocess_options = QComboBox()
    parent.preprocess_options.addItems(["Reducción de ruido", "Ambos"])
    preprocess_layout.addWidget(parent.preprocess_options)

    # Add the horizontal layout to the main vertical layout
    layout.addLayout(preprocess_layout)


    parent.attach_button = QPushButton('Cargar Imagen')
    parent.attach_button.clicked.connect(parent.load_image)
    layout.addWidget(parent.attach_button)

    parent.analyze_button = QPushButton('Analizar Imagen')
    parent.analyze_button.clicked.connect(parent.analyze_image)
    layout.addWidget(parent.analyze_button)

    """parent.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
    parent.sensitivity_slider.setRange(0, 100)
    parent.sensitivity_slider.setValue(50)
    layout.addWidget(QLabel("Sensibilidad del modelo:"))
    layout.addWidget(parent.sensitivity_slider)"""""

  

    parent.history_button = QPushButton('Ver Historial de Pacientes')
    parent.history_button.clicked.connect(parent.load_patient_history)
    layout.addWidget(parent.history_button)

    return layout

def create_right_widget(parent):
    right_widget = QTabWidget()

    # Pestaña de Resultados
    results_tab = QWidget()
    results_layout = QVBoxLayout(results_tab)

    parent.image_viewer = ImageViewer()
    parent.image_viewer.setFixedSize(400, 400)
    results_layout.addWidget(parent.image_viewer)

    parent.result_text = QTextEdit()
    parent.result_text.setReadOnly(True)
    results_layout.addWidget(QLabel("Resultado:"))
    results_layout.addWidget(parent.result_text)

    parent.metrics_table = QTableView()
    results_layout.addWidget(QLabel("Métricas del modelo:"))
    results_layout.addWidget(parent.metrics_table)

    right_widget.addTab(results_tab, "Resultados")

    # Pestaña de Historial
    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    parent.history_list = QListWidget()
    parent.history_list.itemClicked.connect(parent.load_from_history)
    history_layout.addWidget(parent.history_list)

    right_widget.addTab(history_tab, "Historial")

    # Pestaña de Comparación
    comparison_tab = QWidget()
    comparison_layout = QVBoxLayout(comparison_tab)
    parent.comparison_text = QTextEdit()
    parent.comparison_text.setReadOnly(True)
    comparison_layout.addWidget(parent.comparison_text)

    right_widget.addTab(comparison_tab, "Comparación")

    return right_widget
