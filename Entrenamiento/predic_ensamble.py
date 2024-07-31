import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

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

def load_models():
    meta_clf = joblib.load('ensemble_metaclassifier_final.joblib')
    models = {
        'EfficientNetV2B0': load_model('EfficientNetV2B0_cnn_final.h5'),
        'Xception': load_model('Xception_cnn_final.h5'),
        'DenseNet121': load_model('DenseNet121_cnn_final.h5'),
        'ResNet50': load_model('ResNet50_cnn_final.h5'),
        'MobileNet': load_model('MobileNet_cnn_final.h5'),
        'InceptionV3': load_model('InceptionV3_cnn_final.h5')
    }
    return meta_clf, models

def preprocess_image(image_path):
    preprocessor = ImagePreprocessor()
    img_array = preprocessor.preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_metadata(age, sex, localization):
    try:
        sex_encoder = OneHotEncoder(sparse_output=False)
        loc_encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        sex_encoder = OneHotEncoder(sparse=False)
        loc_encoder = OneHotEncoder(sparse=False)
    
    sex_encoder.fit([['male'], ['female']])
    loc_encoder.fit([['scalp'], ['back'], ['chest'], ['face'], ['foot'], ['hand'], ['lower extremity'], ['neck'], ['trunk'], ['upper extremity']])
    
    sex_encoded = sex_encoder.transform([[sex]])
    loc_encoded = loc_encoder.transform([[localization]])
    
    metadata = np.hstack([np.array([[age]]), sex_encoded, loc_encoded])
    
    return metadata.flatten()


def extract_features(models, img_array):
    all_features = []
    for model in models.values():
        feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        features = feature_extractor.predict(img_array)
        all_features.append(features.flatten())
    return np.hstack(all_features)

def predict(image_path, age, sex, localization):
    meta_clf, models = load_models()
    
    img_array = preprocess_image(image_path)
    metadata = preprocess_metadata(age, sex, localization)
    
    features = extract_features(models, img_array)
    
    # Instead of combining features and metadata, keep them separate
    X_features = features.reshape(1, -1)
    X_metadata = metadata.reshape(1, -1)
    
    # Predict using both features and metadata
    prediction = meta_clf.predict(X_features, X_metadata)
    prediction_proba = meta_clf.predict_proba(X_features, X_metadata)
    
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    result = {
        'predicted_class': class_names[prediction[0]],
        'probabilities': dict(zip(class_names, prediction_proba[0]))
    }
    
    return result


if __name__ == "__main__":
    image_path = 'ISIC_0028879.jpg'
    age = 30
    sex = 'male'
    localization = 'back'
    
    result = predict(image_path, age, sex, localization)
    print(f"Clase predicha: {result['predicted_class']}")
    print("Probabilidades:")
    for class_name, prob in result['probabilities'].items():
        print(f"{class_name}: {prob:.4f}")