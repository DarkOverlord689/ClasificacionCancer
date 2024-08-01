import joblib
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging
import cv2
from meta_classifier import MetaClassifier
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

class PredictionSystem:
    def __init__(self, metadata_path):
        self.metadata_features, self.onehot_encoder_sex, self.onehot_encoder_loc = self.load_metadata(metadata_path)
        self.meta_clf, self.models = self.load_models()
        self.feature_extractors = self.create_feature_extractors()
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.image_preprocessor = ImagePreprocessor()

    @staticmethod
    def load_metadata(metadata_path):
        metadata = pd.read_csv(metadata_path)
        
        onehot_encoder_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        sex_encoded = onehot_encoder_sex.fit_transform(metadata[['sex']])
        
        onehot_encoder_loc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        loc_encoded = onehot_encoder_loc.fit_transform(metadata[['localization']])
        
        metadata_features = np.hstack([
            metadata[['age']].values,
            sex_encoded,
            loc_encoded
        ])

        return metadata_features, onehot_encoder_sex, onehot_encoder_loc

    @staticmethod
    def load_models():
        logging.info("Cargando modelos...")
        try:
            meta_clf = joblib.load('ensemble_metaclassifier_final.joblib')
            
            if not isinstance(meta_clf, MetaClassifier):
                raise TypeError("El objeto cargado no es una instancia de MetaClassifier")
            
            models = {
                'EfficientNetV2B0': load_model('best_model_EfficientNetV2B0.h5'),
                'Xception': load_model('best_model_Xception.h5'),
                'DenseNet121': load_model('best_model_DenseNet121.h5'),
                'ResNet50': load_model('best_model_ResNet50.h5'),
                'MobileNet': load_model('best_model_MobileNet.h5'),
                'InceptionV3': load_model('best_model_InceptionV3.h5')
            }
            logging.info("Modelos cargados exitosamente")
            return meta_clf, models
        except Exception as e:
            logging.error(f"Error al cargar los modelos: {str(e)}")
            raise

    def create_feature_extractors(self):
        return {name: Model(inputs=model.input, outputs=model.layers[-2].output) 
                for name, model in self.models.items()}

    def preprocess_image(self, image_path):
        try:
            img = self.image_preprocessor.preprocess_image(image_path)
            img_array = np.expand_dims(img, axis=0)
            return img_array
        except Exception as e:
            logging.error(f"Error al preprocesar la imagen: {str(e)}")
            raise

    def preprocess_metadata(self, age, sex, localization):
        metadata = pd.DataFrame({'age': [age], 'sex': [sex], 'localization': [localization]})
        sex_encoded = self.onehot_encoder_sex.transform(metadata[['sex']])
        loc_encoded = self.onehot_encoder_loc.transform(metadata[['localization']])
        return np.hstack([metadata[['age']].values, sex_encoded, loc_encoded]).flatten()

    def extract_features(self, img_array):
        return np.hstack([extractor.predict(img_array).flatten() for extractor in self.feature_extractors.values()])

    def predict(self, model_name, image_path, age, sex, localization):
        try:
            img_array = self.preprocess_image(image_path)
            metadata = self.preprocess_metadata(age, sex, localization)
            
            features = self.extract_features(img_array)
            
            features = features.reshape(1, -1)
            metadata = metadata.reshape(1, -1)
            
            prediction = self.meta_clf.predict(features, metadata)
            prediction_proba = self.meta_clf.predict_proba(features, metadata)
            
            result = {
                'predicted_class': self.class_names[prediction[0]] if prediction.shape[0] > 0 else "Error en la predicción",
                'probabilities': dict(zip(self.class_names, prediction_proba[0])) if prediction_proba.shape[0] > 0 else {}
            }
            
            return result
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}")
            raise

# Función que será llamada desde el método analyze_image
def predicto(model_name, image_path, age, sex, localization, metadata_path='HAM10000_metadata.csv'):
    prediction_system = PredictionSystem(metadata_path)
    return prediction_system.predict(model_name, image_path, age, sex, localization)
