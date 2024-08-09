import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB2, MobileNet, InceptionV3, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import EfficientNetV2B0
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from meta_classifier import MetaClassifier
import joblib
from sklearn.metrics import classification_report
from utils import calculate_metrics, get_metrics_per_class, plot_confusion_matrix, plot_training_history, plot_metrics, compare_models


def load_data(image_dir, metadata_path):
    metadata = pd.read_csv(metadata_path)
    X = []
    y = []
    metadata_features = []

    onehot_encoder_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sex_encoded = onehot_encoder_sex.fit_transform(metadata[['sex']])
    
    onehot_encoder_loc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    loc_encoded = onehot_encoder_loc.fit_transform(metadata[['localization']])
    
    for index, row in metadata.iterrows():
        img_path = os.path.join(image_dir, row['image_id'] + '.jpg')
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            X.append(img)
            y.append(row['dx'])
            
            metadata_feature = [
                row['age'],  # Mantenemos los NaN aquí
                *sex_encoded[index],
                *loc_encoded[index]
            ]
            metadata_features.append(metadata_feature)

    return np.array(X), np.array(y), np.array(metadata_features)



def preprocess_data(X, y):
    X = X / 255.0  # Normalización
    y_encoded = pd.get_dummies(y).values
    return X, y_encoded

def build_model(model_name, input_shape, num_classes):
    base_models = {
        'DenseNet121': DenseNet121,
        'ResNet50': ResNet50,
        'MobileNet': MobileNet,
        'InceptionV3': InceptionV3,
        'Xception': Xception,
        'EfficientNetV2B0': EfficientNetV2B0
    }
    
    if model_name not in base_models:
        raise ValueError(f"Modelo {model_name} no soportado")
    
    base_model = base_models[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def clear_gpu_memory():
    logging.info("Sesión de Keras limpiada nuevamente.")
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()




def train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights):
    clear_gpu_memory()
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'best_model_{model_name}.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    datagen.fit(X_train)
    clear_gpu_memory()
    logging.info(f"Iniciando entrenamiento de {model_name}...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return history, model

def extract_features(model, X, batch_size=16):
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_features = feature_extractor.predict(batch)
        features.append(batch_features)
    return np.concatenate(features)

from sklearn.metrics import precision_recall_curve, auc, cohen_kappa_score, f1_score, accuracy_score
from scipy.stats import hmean

def adjust_prediction_threshold(y_true, y_pred_proba, class_names):
    n_classes = len(class_names)
    optimal_thresholds = []
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds.append(thresholds[optimal_idx])
    
    return optimal_thresholds

def predict_with_threshold(y_pred_proba, thresholds):
    return (y_pred_proba > thresholds).astype(int)

def calculate_metrics(y_true, y_pred, y_pred_proba, class_names):
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Weighted F1-score
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Macro-averaged AUPRC
    auprc_scores = []
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        auprc_scores.append(auc(recall, precision))
    metrics['macro_auprc'] = np.mean(auprc_scores)
    
    # Harmonic mean of per-class F1-scores
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    metrics['harmonic_mean_f1'] = hmean(per_class_f1)
    
    return metrics


def train_and_evaluate_ensemble(models_to_train, X_train, y_train, X_val, y_val, metadata_train, metadata_val, class_weights, class_names):
    all_train_features = []
    all_val_features = []
    histories = []
    results = {}
    
    for model_name in models_to_train:
        model = build_model(model_name, input_shape=(224, 224, 3), num_classes=y_train.shape[1])
        history, trained_model = train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights)
        histories.append(history)
        
        train_features = extract_features(trained_model, X_train)
        val_features = extract_features(trained_model, X_val)
        
        all_train_features.append(train_features)
        all_val_features.append(val_features)
        
        # Guardar métricas del modelo individual
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        results[model_name] = {'accuracy': final_val_accuracy, 'loss': final_val_loss}
        
        trained_model.save(f'{model_name}_cnn_final.h5')
        logging.info(f"Modelo {model_name} guardado")

        plot_training_history(history, model_name)
        clear_gpu_memory()
    
    X_train_cnn = np.hstack(all_train_features)
    X_val_cnn = np.hstack(all_val_features)
    
    meta_clf = MetaClassifier(classifier_type='auto', n_jobs=-1)
    meta_clf.fit(X_train_cnn, metadata_train, np.argmax(y_train, axis=1))
    
    y_pred_proba = meta_clf.predict_proba(X_val_cnn, metadata_val)
    y_true = np.argmax(y_val, axis=1)
    
    # Ajuste de umbrales
    optimal_thresholds = adjust_prediction_threshold(y_true, y_pred_proba, class_names)
    y_pred = predict_with_threshold(y_pred_proba, optimal_thresholds)
    
    # Asegurarse de que y_pred sea un array 1D de etiquetas de clase
    y_pred_class = np.argmax(y_pred, axis=1)
    
    # Cálculo de métricas
    metrics = calculate_metrics(y_true, y_pred_class, y_pred_proba, class_names)
    
    print("Métricas para el Ensemble con MetaClassifier:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nReporte de clasificación para el Ensemble con MetaClassifier:")
    print(classification_report(y_true, y_pred_class, target_names=class_names))
    
    plot_confusion_matrix(y_true, y_pred_class, class_names, "Ensemble")
    
    joblib.dump(meta_clf, 'ensemble_metaclassifier_final.joblib')
    logging.info("MetaClassifier del Ensemble guardado")
    
    class_metrics = get_metrics_per_class(y_true, y_pred_class, class_names)
    
    # Guardar las métricas del ensemble final
    results['Ensemble'] = {'accuracy': metrics['accuracy'], 'loss': 1 - metrics['weighted_f1']}
    
    # Comparar todos los modelos
    compare_models(results)
    
    return class_metrics, metrics, optimal_thresholds