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
import os
from datetime import datetime
from utils import get_patient_directory, get_interpretation_directory
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from PIL import Image, ImageDraw, ImageFont
import io



print("Directorio actual:", os.getcwd())
print("Contenido del directorio:", os.listdir())

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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Aplicar la transformada Black Hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Intensificar los contornos del vello
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Generar una máscara
        mask = thresh.copy()
        
        # Realizar inpainting
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Redimensionar la imagen a 224x224 píxeles
        result = cv2.resize(result, (224, 224))
      
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
        except ImportError as ie:
            logging.error(f"Error de importación: {str(ie)}")
        except Exception as e:
            logging.error(f"Error al cargar los modelos: {str(e)}")


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

  

    
    
    def interpret_prediction(self, image_path, predicted_class):
        img_array = self.preprocess_image(image_path)
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        class_index = self.class_names.index(predicted_class)
        
        heatmaps = {}
        for model_name in self.models.keys():
            heatmap = self.generate_gradcam(img_array, model_name, class_index)
            heatmap_img = self.apply_heatmap(original_img, heatmap)
            annotated_img = self.add_annotations(heatmap_img, model_name, predicted_class)
            heatmaps[model_name] = self.pil_to_bytes(annotated_img)
        
        explanation = self.generate_explanation(predicted_class)
        heatmaps['explanation'] = explanation
        
        return heatmaps

    def apply_heatmap(self, img, heatmap):
        heatmap = np.uint8(255 * heatmap)
        jet = plt.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = np.array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + img
        return Image.fromarray(np.uint8(superimposed_img))

    def add_annotations(self, img, model_name, predicted_class):
        # Crear una nueva capa para el área semi-transparente
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Convertir la imagen original a RGBA si no lo está
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Calcular dimensiones del recuadro
        box_height = 120  # Altura del recuadro
        box_y = img.height - box_height  # Posición Y del recuadro
        
        # Dibujar un recuadro semi-transparente negro
        draw.rectangle([0, box_y, img.width, img.height], 
                    fill=(0, 0, 0, 180))  # El último valor (180) controla la transparencia
        
        # Combinar la capa del recuadro con la imagen original
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        
        # Usar una fuente más grande si está disponible
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Agregar título
        draw.text((10, 10), f"Model: {model_name}", font=font, fill=(255, 255, 255))
        draw.text((10, 30), f"Predicted: {predicted_class}", font=font, fill=(255, 255, 255))
        
        # Agregar leyenda en el recuadro negro
        legend_start_y = img.height - 100
        draw.rectangle([10, legend_start_y, 30, legend_start_y + 20], fill='red')
        draw.text((35, legend_start_y), "High activation", font=font, fill=(255, 255, 255))
        
        draw.rectangle([10, legend_start_y + 30, 30, legend_start_y + 50], fill='blue')
        draw.text((35, legend_start_y + 30), "Low activation", font=font, fill=(255, 255, 255))
        
        # Agregar explicación en la parte superior
        draw.text((10, 50), "Red areas indicate", font=font, fill=(255, 255, 255))
        draw.text((10, 70), "regions of interest", font=font, fill=(255, 255, 255))
        draw.text((10, 90), "for the prediction", font=font, fill=(255, 255, 255))
        
        return img
    def pil_to_bytes(self, img):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def generate_explanation(self, predicted_class):
        explanation = f"Diagnosis: {predicted_class}\n\n"
        
        if predicted_class == "melanoma":
            explanation += ("El modelo detectó características comúnmente asociadas con el melanoma, como"
                            "bordes irregulares, patrones asimétricos o variaciones de color."
                            "Estas áreas están resaltadas en rojo en los mapas de calor.\n\n")
        elif predicted_class == "basal cell carcinoma":
            explanation += ("Los mapas de calor muestran áreas concentradas de activación, que pueden indicar"
                            "la presencia de una lesión bien definida con un borde nacarado o ceroso, "
                            "característico del carcinoma de células basales.\n\n")
        elif predicted_class == "squamous cell carcinoma":
            explanation += ("El modelo destacó áreas que pueden representar superficies escamosas o costrosas, "
                            "que suelen observarse en el carcinoma de células escamosas. "
                            "Los mapas de calor muestran patrones de activación difusos en toda el área de la lesión.\n\n")
        
        explanation += ("Tenga en cuenta que, si bien estas visualizaciones brindan información sobre el proceso de toma de decisiones del modelo "
                        "no deben considerarse como un diagnóstico definitivo. "
                        "Siempre consulte con un profesional de la salud calificado para obtener asesoramiento y diagnóstico médicos precisos. ")
        
        return explanation
    


def predicto(model_name, image_path, age, sex, localization, patient_id, metadata_path='metadatos_T.csv'):
    prediction_system = PredictionSystem(metadata_path)
    result = prediction_system.predict(model_name, image_path, age, sex, localization)
    
    # Obtener la fecha y hora actual para el diagnóstico
    diagnosis_date = datetime.now()
    
    # Crear el directorio base de interpretaciones si no existe
    base_interpretation_dir = 'interpretations'
    os.makedirs(base_interpretation_dir, exist_ok=True)
    
    # Guardar las imágenes de interpretación
    interpretations = {}
    for model_name, heatmap_bytes in result['interpretations'].items():
        if model_name != 'explanation':
            try:
                # Usar el mismo nombre de archivo para cada modelo, sobrescribiendo el anterior
                image_filename = f'{model_name}_interpretation.png'
                image_path = os.path.join(base_interpretation_dir, image_filename)
                
                # Guardar/sobrescribir la imagen
                with open(image_path, 'wb') as f:
                    f.write(heatmap_bytes)
                
                interpretations[model_name] = image_path
                logging.info(f"Saved/updated interpretation image for {model_name}.")
            except Exception as e:
                logging.error(f"Error saving interpretation image for {model_name}: {str(e)}")
    
    # Guardar la explicación en texto
    explanation_filename = 'explanation.txt'
    explanation_path = os.path.join(base_interpretation_dir, explanation_filename)
    with open(explanation_path, 'w') as f:
        f.write(result['interpretations']['explanation'])
    
    interpretations['explanation'] = explanation_path
    
    # Actualizar el resultado con las rutas de las interpretaciones
    result['interpretations'] = interpretations
    
    # Agregar información adicional al resultado
    result['patient_id'] = patient_id
    result['diagnosis_date'] = diagnosis_date.strftime("%Y-%m-%d %H:%M:%S")
    result['interpretation_dir'] = base_interpretation_dir
    
    return result
