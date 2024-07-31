import os
import logging
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

class ImagePreprocessor:
    def preprocess_image(self, image_path, size=(224, 224)):
        img = cv2.imread(image_path)
        img = self.remove_hair_enhanced(img)
        img = cv2.resize(img, size)
        img = self.normalize_pixel_data(img)
        return img

    def remove_hair_enhanced(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create a hair mask using black hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Enhance the hair mask
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(thresh, kernel, iterations=1)
        
        # Inpaint
        inpainted_img = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)
        
        # Apply bilateral filter to preserve edges
        smooth = cv2.bilateralFilter(inpainted_img, 9, 75, 75)
        
        # Combine original and smoothed image
        result = cv2.addWeighted(img, 0.7, smooth, 0.3, 0)
        
        return result

    def resize_image(self, img, size):
        img = Image.fromarray(img)
        resample_method = getattr(Image, 'LANCZOS', Image.BICUBIC)
        img = img.resize(size, resample_method)
        return np.array(img)

    def normalize_pixel_data(self, img):
        return img.astype('float32') / 255.0

def process_and_store_images(image_dir, metadata_path, processed_image_dir):
    if not os.path.exists(processed_image_dir):
        os.makedirs(processed_image_dir)

    metadata = pd.read_csv(metadata_path)
    preprocessor = ImagePreprocessor()
    
    for index, row in metadata.iterrows():
        raw_img_path = os.path.join(image_dir, row['image_id'] + '.jpg')
        processed_img_path = os.path.join(processed_image_dir, row['image_id'] + '.jpg')
        
        if not os.path.exists(raw_img_path):
            logging.warning(f"Archivo no encontrado: {raw_img_path}")
            continue
        
        try:
            if os.path.exists(processed_img_path):
                logging.info(f"Imagen preprocesada ya existe: {processed_img_path}")
                continue
            else:
                img = preprocessor.preprocess_image(raw_img_path)
                img = (img * 255).astype(np.uint8)  # Convertir de vuelta a uint8
                cv2.imwrite(processed_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                logging.info(f"Imagen preprocesada y guardada: {processed_img_path}")
        except Exception as e:
            logging.error(f"Error al procesar la fila {index}: {e}")
            continue

if __name__ == "__main__":
    image_dir = "HAM"
    metadata_path = "HAM10000_metadata.csv"
    processed_image_dir = "LLL"
    
    process_and_store_images(image_dir, metadata_path, processed_image_dir)