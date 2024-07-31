# image_preprocessor.py

import numpy as np
from PIL import Image, ImageEnhance

class ImagePreprocessor:
    def adjust_white_balance(self, img):
        """
        Ajusta el balance de blancos de la imagen.
        
        :param img: Imagen de entrada
        :return: Imagen con balance de blancos ajustado
        """
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)  # Ajustar el factor según sea necesario
        return np.array(img)

    def resize_image(self, img, size):
        """
        Redimensiona la imagen a un tamaño específico.
        
        :param img: Imagen de entrada
        :param size: Nueva dimensión (ancho, alto)
        :return: Imagen redimensionada
        """
        img = Image.fromarray(img)
        resample_method = getattr(Image, 'LANCZOS', Image.BICUBIC)
        img = img.resize(size, resample_method)
        return np.array(img)

    def normalize_pixel_data(self, img):
        """
        Normaliza los datos de los píxeles de la imagen.
        
        :param img: Imagen de entrada
        :return: Imagen normalizada
        """
        img = img.astype('float32') / 255.0  # Escalar los valores de 0 a 1
        return img
