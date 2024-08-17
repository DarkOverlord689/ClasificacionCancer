"""
    Este módulo contiene el entrenamiento del modelo, aqui se realiza la predicción de los modelos individuales y del metaclasificador
    este hace el llamado de dos modulos auxiliares, el metaclasificador y el utils

    sigue un pipeline establecido en el modulo principal app
    
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB2, MobileNet, InceptionV3, VGG16, Xception, EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from meta_classifier import MetaClassifier
import joblib
from sklearn.metrics import classification_report, precision_recall_curve, auc, cohen_kappa_score, f1_score, accuracy_score
from scipy.stats import hmean
from utils import calculate_metrics, get_metrics_per_class, plot_confusion_matrix, plot_training_history, plot_metrics, compare_models

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(image_dir, metadata_path):
    """
    Carga los datos de imágenes y metadatos.
    
    Parámetros:
    - image_dir: Directorio que contiene las imágenes
    - metadata_path: Ruta al archivo CSV de metadatos
    
    Retorna:
    - X: Array de imágenes
    - y: Array de etiquetas
    - metadata_features: Array de características de metadatos
    """
    metadata = pd.read_csv(metadata_path)
    X, y, metadata_features = [], [], []

    # Codificación one-hot para características categóricas
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
                row['age'],
                *sex_encoded[index],
                *loc_encoded[index]
            ]
            metadata_features.append(metadata_feature)

    return np.array(X), np.array(y), np.array(metadata_features)

def preprocess_data(X, y):
    """
    Preprocesa los datos normalizando las imágenes y codificando las etiquetas.
    
    Parámetros:
    - X: Array de imágenes
    - y: Array de etiquetas
    
    Retorna:
    - X: Array de imágenes normalizadas
    - y_encoded: Array de etiquetas codificadas en one-hot
    """
    X = X / 255.0  # Normalización
    y_encoded = pd.get_dummies(y).values
    return X, y_encoded

def build_model(model_name, input_shape, num_classes):
    """
    Construye un modelo de CNN basado en una arquitectura preentrenada.
    
    Parámetros:
    - model_name: Nombre del modelo base a utilizar
    - input_shape: Forma de entrada de las imágenes
    - num_classes: Número de clases para la capa de salida
    
    Retorna:
    - model: Modelo de Keras compilado
    """
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
    """
    Limpia la memoria de la GPU.
    """
    logging.info("Sesión de Keras limpiada nuevamente.")
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

def train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights):
    """
    Entrena un modelo de CNN con fine-tuning.
    
    Parámetros:
    - model: Modelo de Keras a entrenar
    - model_name: Nombre del modelo (para logging)
    - X_train, y_train: Datos de entrenamiento
    - X_val, y_val: Datos de validación
    - class_weights: Pesos de las clases para manejar desbalance
    
    Retorna:
    - history: Historial de entrenamiento
    - model: Modelo entrenado
    """
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
    train_generator = datagen.flow(X_train, y_train, batch_size=8)
    
    # Entrenamiento inicial
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 16,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Fine-tuning
    for layer in model.layers[-20:]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history_fine_tune = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        steps_per_epoch=len(X_train) // 16,
        validation_data=(X_val, y_val),
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Combinar historiales de entrenamiento
    history = history_initial
    for key in history.history:
        history.history[key].extend(history_fine_tune.history[key])
    history.epoch.extend([max(history.epoch) + 1 + e for e in history_fine_tune.epoch])
    
    return history, model

def extract_features(model, X, batch_size=16):
    """
    Extrae características de las imágenes usando el modelo entrenado.
    
    Parámetros:
    - model: Modelo de Keras entrenado
    - X: Array de imágenes
    - batch_size: Tamaño del lote para la extracción
    
    Retorna:
    - features: Array de características extraídas
    """
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_features = feature_extractor.predict(batch)
        features.append(batch_features)
    return np.concatenate(features)

def adjust_prediction_threshold(y_true, y_pred_proba, class_names):
    """
    Ajusta los umbrales de predicción para cada clase.
    
    Parámetros:
    - y_true: Etiquetas verdaderas
    - y_pred_proba: Probabilidades predichas
    - class_names: Nombres de las clases
    
    Retorna:
    - optimal_thresholds: Lista de umbrales óptimos para cada clase
    """
    n_classes = len(class_names)
    optimal_thresholds = []
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds.append(thresholds[optimal_idx])
    
    return optimal_thresholds

def predict_with_threshold(y_pred_proba, thresholds):
    """
    Realiza predicciones usando umbrales personalizados.
    
    Parámetros:
    - y_pred_proba: Probabilidades predichas
    - thresholds: Umbrales para cada clase
    
    Retorna:
    - Predicciones binarias basadas en los umbrales
    """
    return (y_pred_proba > thresholds).astype(int)

def calculate_metrics(y_true, y_pred, y_pred_proba, class_names):
    """
    Calcula métricas de rendimiento del modelo.
    
    Parámetros:
    - y_true: Etiquetas verdaderas
    - y_pred: Predicciones del modelo
    - y_pred_proba: Probabilidades predichas
    - class_names: Nombres de las clases
    
    Retorna:
    - metrics: Diccionario con las métricas calculadas
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    auprc_scores = []
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        auprc_scores.append(auc(recall, precision))
    metrics['macro_auprc'] = np.mean(auprc_scores)
    
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    metrics['harmonic_mean_f1'] = hmean(per_class_f1)
    
    return metrics




def train_and_evaluate_ensemble(models_to_train, X_train, y_train, X_val, y_val, metadata_train, metadata_val, class_weights, class_names):
    
    """
    Entrena y evalúa un ensemble de modelos de CNN.
    
    Parámetros:
    - models_to_train: Lista de nombres de modelos a entrenar
    - X_train, y_train: Datos de entrenamiento
    - X_val, y_val: Datos de validación
    - metadata_train, metadata_val: Metadatos de entrenamiento y validación
    - class_weights: Pesos de las clases
    - class_names: Nombres de las clases
    
    Retorna:
    - class_metrics: Métricas por clase
    - metrics: Métricas generales
    - optimal_thresholds: Umbrales óptimos por clase
    """
    
    all_train_features = []
    all_val_features = []
    histories = []
    results = {}
    
    # Entrenamiento de modelos individuales
    for model_name in models_to_train:
        model = build_model(model_name, input_shape=(224, 224, 3), num_classes=y_train.shape[1])
        history, trained_model = train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights)
        histories.append(history)
        
        # Extracción de características
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
    # Preparación de datos para el meta-clasificador
    X_train_cnn = np.hstack(all_train_features)
    X_val_cnn = np.hstack(all_val_features)
    

    # Entrenamiento del meta-clasificador
    meta_clf = MetaClassifier(classifier_type='auto', n_jobs=-1)
    meta_clf.fit(X_train_cnn, metadata_train, np.argmax(y_train, axis=1))
    
    # Predicciones y ajuste de umbrales
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
