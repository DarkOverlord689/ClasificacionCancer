#---------------librerias ----------------------------------------------
import joblib
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging
import cv2
from meta_classifier import MetaClassifier
import tensorflow as tf
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Esta clase maneja el procesamiento de imágenes antes de ser ingresadas a los modelos de redes neuronales
"""
class ImagePreprocessor:
    def preprocess_image(self, image_path, size=(224, 224)):

        """
        Preprocesa una imagen, realiza la eliminación de vello y normalización.
        Args:
            image_path (str): Ruta de la imagen a procesar.
            size (tuple): Tamaño al cual redimensionar la imagen.
        Returns:
            np.array: Imagen preprocesada.
        """
        img = cv2.imread(image_path)
        img = self.remove_hair_enhanced(img)
        img = cv2.resize(img, size)
        img = self.normalize_pixel_data(img)
        return img

    def remove_hair_enhanced(self, img):

        """
        Aplica procesamiento de imagen para eliminar vello de las imágenes médicas.
        Args:
            img (np.array): Imagen en formato numpy array.
        Returns:
            np.array: Imagen con el vello eliminado.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convierte la imagen a escala de grises
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # Detecta sombras del vello
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)  # Genera una máscara binaria del vello
        kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(thresh, kernel, iterations=1)  # Dilata la máscara para cubrir áreas de vello
        inpainted_img = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)  # Restaura la imagen inpaintando el área del vello
        smooth = cv2.bilateralFilter(inpainted_img, 9, 75, 75)  # Suaviza la imagen resultante
        result = cv2.addWeighted(img, 0.7, smooth, 0.3, 0)  # Combinación de la imagen suavizada y original
        return result

    def normalize_pixel_data(self, img):
        """
        Normaliza los valores de los píxeles de la imagen entre 0 y 1.
        Args:
            img (np.array): Imagen de entrada.
        Returns:
            np.array: Imagen normalizada.
        """
        return img.astype('float32') / 255.0  # Normaliza dividiendo por 255

"""Esta clase maneja el flujo de trabajo completo para la predicción, desde el preprocesamiento de imágenes y 
metadatos hasta la predicción del modelo y la interpretación de los resultados."""

class PredictionSystem:
    def __init__(self, metadata_path):
        """
        Constructor del sistema de predicción, inicializa los modelos, clasificadores, y preprocessors.
        Args:
            metadata_path (str): Ruta del archivo de metadatos.
        """
        self.metadata_features, self.onehot_encoder_sex, self.onehot_encoder_loc = self.load_metadata(metadata_path)  # Carga metadatos
        self.meta_clf, self.models = self.load_models()  # Carga los modelos entrenados y metaclassifier
        self.feature_extractors = self.create_feature_extractors()  # Crea extractores de características
        self.class_names = ['basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']  # Nombres de las clases
        self.image_preprocessor = ImagePreprocessor()  # Inicializa el preprocesador de imágenes
        # Umbrales para aplicar en la predicción
        self.thresholds = {
            'basal cell carcinoma': 0.3111,
            'melanoma':  0.6740,
            'squamous cell carcinoma': 0.1676
        }

    @staticmethod
    def load_metadata(metadata_path):
        """
        Carga y procesa los metadatos para la predicción.
        Args:
            metadata_path (str): Ruta del archivo CSV con los metadatos.
        Returns:
            tuple: Características de metadatos, codificadores one-hot para sexo y localización.
        """
        metadata = pd.read_csv(metadata_path)  # Carga el archivo de metadatos
        
        # Codificación one-hot de la columna 'sex'
        onehot_encoder_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        sex_encoded = onehot_encoder_sex.fit_transform(metadata[['sex']])
        
        # Codificación one-hot de la columna 'localization'
        onehot_encoder_loc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        loc_encoded = onehot_encoder_loc.fit_transform(metadata[['localization']])
        
        # Concatenar las características de edad, sexo y localización
        metadata_features = np.hstack([
            metadata[['age']].values,
            sex_encoded,
            loc_encoded
        ])

        return metadata_features, onehot_encoder_sex, onehot_encoder_loc

    @staticmethod
    def load_models():
        """
        Carga los modelos entrenados y el MetaClassifier.
        Returns:
            tuple: MetaClassifier y modelos CNN preentrenados.
        """
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
        """
        Crea modelos para la extracción de características a partir de las capas penúltimas de las CNN.
        Returns:
            dict: Modelos de extracción de características para cada CNN.
        """
        return {name: Model(inputs=model.input, outputs=model.layers[-2].output) 
                for name, model in self.models.items()}

    def preprocess_image(self, image_path):
        """
        Preprocesa la imagen antes de realizar predicciones.
        Args:
            image_path (str): Ruta de la imagen a preprocesar.
        Returns:
            np.array: Imagen preprocesada lista para predicción.
        """
        try:
            img = self.image_preprocessor.preprocess_image(image_path)
            img_array = np.expand_dims(img, axis=0)
            return img_array
        except Exception as e:
            logging.error(f"Error al preprocesar la imagen: {str(e)}")
            raise

    def preprocess_metadata(self, age, sex, localization):
       """
       Preprocesa los metadatos (edad, sexo y localización) para usarlos en la predicción.
       Args:
           age (int): Edad del paciente.
           sex (str): Sexo del paciente.
           localization (str): Localización de la lesión.
       Returns:
           np.array: Metadatos procesados y codificados.
       """
       metadata = pd.DataFrame({'age': [age], 'sex': [sex], 'localization': [localization]})
       sex_encoded = self.onehot_encoder_sex.transform(metadata[['sex']])
       loc_encoded = self.onehot_encoder_loc.transform(metadata[['localization']])
       preprocessed_metadata = np.hstack([metadata[['age']].values, sex_encoded, loc_encoded]).flatten()
       logging.info(f"Preprocessed metadata shape: {preprocessed_metadata.shape}")
       return preprocessed_metadata

    def extract_features(self, img_array):
       """
       Extrae las características de la imagen utilizando los modelos de CNN.
       Args:
           img_array (np.array): Imagen preprocesada.
       Returns:
           np.array: Características extraídas de cada modelo.
       """
       features = []
       for name, extractor in self.feature_extractors.items():
           feature = extractor.predict(img_array).flatten()
           logging.info(f"Features extracted from {name}: {feature.shape}")
           features.append(feature)
       combined_features = np.hstack(features)
       logging.info(f"Combined features shape: {combined_features.shape}")
       return combined_features

    def predict(self, model_name, image_path, age, sex, localization):
        try:
            img_array = self.preprocess_image(image_path)
            metadata = self.preprocess_metadata(age, sex, localization)
            
            features = self.extract_features(img_array)
            
            features = features.reshape(1, -1)
            metadata = metadata.reshape(1, -1)
            
            logging.info(f"Features shape: {features.shape}")
            logging.info(f"Metadata shape: {metadata.shape}")
            
            # Pass features and metadata separately to predict_proba
            prediction_proba = self.meta_clf.predict_proba(features, metadata)[0]
            
            # Aplicar umbrales
            predicted_class = None
            max_prob = 0
            for i, class_name in enumerate(self.class_names):
                if prediction_proba[i] > self.thresholds[class_name] and prediction_proba[i] > max_prob:
                    predicted_class = class_name
                    max_prob = prediction_proba[i]
            
            if predicted_class is None:
                predicted_class = "No se pudo clasificar con confianza"

            result = {
                'predicted_class': predicted_class,
                'probabilities': dict(zip(self.class_names, prediction_proba))
            }

            # Generar interpretación
            heatmaps = self.interpret_prediction(image_path, predicted_class)
            result['interpretations'] = heatmaps

            return result
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}")
            raise

    def generate_gradcam(self, img_array, model_name, pred_index):
        model = self.models[model_name]
        last_conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D))
        
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def apply_heatmap(self, img, heatmap):
        heatmap = np.uint8(255 * heatmap)
        jet = plt.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        return superimposed_img

    def interpret_prediction(self, image_path, predicted_class):
        img_array = self.preprocess_image(image_path)
        original_img = plt.imread(image_path)
        
        class_index = self.class_names.index(predicted_class)
        
        heatmaps = {}
        for model_name in self.models.keys():
            heatmap = self.generate_gradcam(img_array, model_name, class_index)
            heatmap_img = self.apply_heatmap(original_img, heatmap)
            heatmaps[model_name] = heatmap_img
        
        return heatmaps
    
    
# Función que será llamada desde el método analyze_image
def predicto(model_name, image_path, age, sex, localization, metadata_path='metadatos_T.csv'):
    prediction_system = PredictionSystem(metadata_path)
    result = prediction_system.predict(model_name, image_path, age, sex, localization)
    # Guardar las imágenes de interpretación
    for model_name, heatmap in result['interpretations'].items():
        heatmap.save(f'interpretation_{model_name}.jpg')
    
    # Eliminar las imágenes del resultado para evitar problemas de serialización
    del result['interpretations']
    return result
