#Contiene funciones para el preprocesamiento de imágenes.
import cv2
import numpy as np

def preprocess_image(image_path, operations):
    img = cv2.imread(image_path)
    
    if 'noise_reduction' in operations:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    if 'normalize' in operations:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    return img

class ImagePreprocessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
    
    def reduce_noise(self):
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
    
    def normalize(self):
        self.image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
    
    def get_processed_image(self):
        return self.image