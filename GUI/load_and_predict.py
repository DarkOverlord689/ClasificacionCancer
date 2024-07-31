import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImagePreprocessor:
    def preprocess_image(self, image_path, size=(224, 224)):
        img = cv2.imread(image_path)
        img = self.remove_hair_enhanced(img)
        img = cv2.resize(img, size)
        img = self.normalize_pixel_data(img)
        return img

    def remove_hair_enhanced(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(thresh, kernel, iterations=1)
        inpainted_img = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)
        smooth = cv2.bilateralFilter(inpainted_img, 9, 75, 75)
        result = cv2.addWeighted(img, 0.7, smooth, 0.3, 0)
        return result

    def normalize_pixel_data(self, img):
        return img.astype('float32') / 255.0

def load_trained_model(model_path):
    """Carga el modelo entrenado desde el archivo."""
    mm='best_model_DenseNet121.h5'
    return tf.keras.models.load_model(mm)

def predict(model, image_path, preprocessor):
    """Realiza la predicción usando el modelo cargado y el preprocesador."""
    processed_image = preprocessor.preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = preprocess_input(processed_image)
    
    prediction = model.predict(processed_image)
    
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    result = {
        'predicted_class': class_names[np.argmax(prediction[0])],
        'probabilities': dict(zip(class_names, prediction[0]))
    }
    
    return result
