from PyQt6.QtWidgets import (QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QSlider, QTabWidget, QTextEdit, QListWidget, QTableView, QWidget,
                             QLineEdit, QFormLayout, QScrollArea, QApplication)
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt6.QtCore import Qt
from image_viewer import ImageViewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

def create_left_layout(parent):
    layout = QVBoxLayout()
    layout.setSpacing(15)  # Aumenta el espacio entre widgets

    # Estilo general
    app = QApplication.instance()
    app.setStyle("Fusion")
    
    # Paleta de colores para entornos clínicos
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    # CAMBIO: Añadir colores para texto y placeholder
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(100, 100, 100))
    app.setPalette(palette)

    # Fuentes
    title_font = QFont("Arial", 24, QFont.Weight.Bold)
    label_font = QFont("Arial", 12)
    button_font = QFont("Arial", 12, QFont.Weight.Bold)

    # Logo y título
    title_label = QLabel('Clasificador multiclase de Cáncer de piel')
    title_label.setFont(title_font)
    layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
    
    logo_label = QLabel()
    logo_pixmap = QPixmap('logo.png')
    if not logo_pixmap.isNull():
        logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
    else:
        logo_label.setText("Logo not found")
    layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

    # Formulario de datos del paciente
    form_widget = QWidget()
    form_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; padding: 10px;")
    form_layout = QFormLayout(form_widget)
    form_layout.setSpacing(10)
    
    fields = [
        ("Nombre:", "name_input"),
        ("Identificación:", "id_input"),
        ("Edad:", "age_input"),
        ("Sexo:", "sex_input", ["Masculino", "Femenino", "Otro"]),
        ("Localización:", "location_input", ["Cabeza", "Cuello", "Tronco", "Brazos", "Piernas", "Otro"])
    ]

    for label_text, attr_name, *options in fields:
        label = QLabel(label_text)
        label.setFont(label_font)
        
        if options:
            widget = QComboBox()
            widget.addItems(options[0])
            widget.setFont(label_font)
        else:
            widget = QLineEdit()
            widget.setFont(label_font)
        
        setattr(parent, attr_name, widget)
        form_layout.addRow(label, widget)

    # CAMBIO: Aplicar estilo al formulario
    input_style = """
    QLineEdit, QTextEdit, QComboBox {
        color: black;
        background-color: white;
        selection-color: white;
        selection-background-color: #0078d7;
    }
    """
    form_widget.setStyleSheet(input_style)

    layout.addWidget(form_widget)

    # Preprocesamiento
    preprocess_layout = QHBoxLayout()
    preprocess_label = QLabel("Preprocesamiento:")
    preprocess_label.setFont(label_font)
    preprocess_layout.addWidget(preprocess_label)
    parent.preprocess_options = QComboBox()
    parent.preprocess_options.addItems(["Reducción de ruido", "Ambos"])
    parent.preprocess_options.setFont(label_font)
    preprocess_layout.addWidget(parent.preprocess_options)
    layout.addLayout(preprocess_layout)

    # Botones
    button_style = """
    QPushButton {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QPushButton:pressed {
        background-color: #3e8e41;
    }
    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
    }
    """
    # Añadir herramientas de edición de imagen
    layout.addWidget(QLabel("Zoom:"))
    parent.zoom_slider = QSlider(Qt.Orientation.Horizontal)
    parent.zoom_slider.setRange(10, 200)
    parent.zoom_slider.setValue(100)
    parent.zoom_slider.valueChanged.connect(parent.update_zoom)
    layout.addWidget(parent.zoom_slider)

    layout.addWidget(QLabel("Contraste:"))
    parent.contrast_slider = QSlider(Qt.Orientation.Horizontal)
    parent.contrast_slider.setRange(50, 150)
    parent.contrast_slider.setValue(100)
    parent.contrast_slider.valueChanged.connect(parent.update_contrast)
    layout.addWidget(parent.contrast_slider)

    layout.addWidget(QLabel("Brillo:"))
    parent.brightness_slider = QSlider(Qt.Orientation.Horizontal)
    parent.brightness_slider.setRange(50, 150)
    parent.brightness_slider.setValue(100)
    parent.brightness_slider.valueChanged.connect(parent.update_brightness)
    layout.addWidget(parent.brightness_slider)

    #Cargar imagen 
    parent.attach_button = QPushButton('Cargar Imagen')
    parent.attach_button.setStyleSheet(button_style)
    parent.attach_button.setFont(button_font)
    parent.attach_button.clicked.connect(parent.load_image)
    parent.attach_button.setToolTip("Haz clic para cargar una imagen del paciente")
    layout.addWidget(parent.attach_button)

    parent.analyze_button = QPushButton('Analizar Imagen')
    parent.analyze_button.setStyleSheet(button_style)
    parent.analyze_button.setFont(button_font)
    parent.analyze_button.clicked.connect(parent.analyze_image)
    parent.analyze_button.setToolTip("Haz clic para analizar la imagen cargada")
    #parent.analyze_button.setEnabled(False)  # Deshabilitado hasta que se cargue una imagen
    layout.addWidget(parent.analyze_button)

    parent.history_button = QPushButton('Ver Historial de Pacientes')
    parent.history_button.setStyleSheet(button_style)
    parent.history_button.setFont(button_font)
    parent.history_button.clicked.connect(parent.load_patient_history)
    parent.history_button.setToolTip("Haz clic para ver el historial de pacientes")
    layout.addWidget(parent.history_button)

    return layout

def create_right_widget(parent):
    right_widget = QTabWidget()
    right_widget.setFont(QFont("Arial", 12))

    # Pestaña de Resultados
    results_tab = QWidget()
    results_layout = QVBoxLayout(results_tab)
    results_layout.setSpacing(10)

    parent.image_viewer = ImageViewer()
    parent.image_viewer.setFixedSize(450, 450)
    results_layout.addWidget(parent.image_viewer)

    parent.result_text = QTextEdit()
    parent.result_text.setReadOnly(True)
    parent.result_text.setFont(QFont("Arial", 12))
    # CAMBIO: Aplicar estilo al result_text
    parent.result_text.setStyleSheet("QTextEdit { color: black; background-color: white; }")
    results_layout.addWidget(QLabel("Resultado:"))
    results_layout.addWidget(parent.result_text)

    """parent.metrics_table = QTableView()
    # CAMBIO: Aplicar estilo al metrics_table
    parent.metrics_table.setStyleSheet("QTableView { border: 1px solid #cccccc; color: black; }")
    results_layout.addWidget(QLabel("Métricas del modelo:"))
    results_layout.addWidget(parent.metrics_table)"""

    right_widget.addTab(results_tab, "Resultados")

    # Pestaña de Historial
    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    parent.history_list = QListWidget()
    parent.history_list.setFont(QFont("Arial", 12))
    parent.history_list.itemClicked.connect(parent.load_from_history)
    # CAMBIO: Aplicar estilo al history_list
    parent.history_list.setStyleSheet("QListWidget { color: black; }")
    history_layout.addWidget(parent.history_list)

    right_widget.addTab(history_tab, "Historial")

    # Pestaña de Comparación
    comparison_tab = QWidget()
    comparison_layout = QVBoxLayout(comparison_tab)
    parent.comparison_text = QTextEdit()
    parent.comparison_text.setReadOnly(True)
    parent.comparison_text.setFont(QFont("Arial", 12))
    # CAMBIO: Aplicar estilo al comparison_text
    parent.comparison_text.setStyleSheet("QTextEdit { color: black; background-color: white; }")
    comparison_layout.addWidget(parent.comparison_text)

    right_widget.addTab(comparison_tab, "Comparación")

    return right_widget
