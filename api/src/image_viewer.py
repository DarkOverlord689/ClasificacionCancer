#Maneja la visualización y manipulación de imágenes.
from PyQt6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QBuffer, QIODevice
from PIL import Image, ImageEnhance
import io

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QPixmap()
        self.zoom_factor = 1
        self.contrast = 1
        self.brightness = 1
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

    def set_image(self, pixmap):
        self.image = pixmap
        self.update_image()

    def update_zoom(self, value):
        self.zoom_factor = value / 100
        self.update_image()

    def update_contrast(self, value):
        self.contrast = value / 100
        self.update_image()

    def update_brightness(self, value):
        self.brightness = value / 100
        self.update_image()

    def update_image(self):
        if self.image.isNull():
            return

        img = self.image.toImage()
        img = img.convertToFormat(QImage.Format.Format_ARGB32)

        # Aplicar zoom
        size = self.image.size() * self.zoom_factor
        img = img.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Convertir QImage a PIL Image para ajustar contraste y brillo
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        img.save(buffer, "PNG")
        pil_img = Image.open(io.BytesIO(buffer.data()))

        # Ajustar contraste y brillo
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(self.contrast)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(self.brightness)

        # Convertir de vuelta a QPixmap
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        pil_img.save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.data())

        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())