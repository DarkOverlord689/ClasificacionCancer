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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from meta_classifier import MetaClassifier
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import EfficientNetV2B0


# Configuración de logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU configurada exitosamente.")
    except RuntimeError as e:
        logging.error(f"Error al configurar GPU: {e}")

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


def clear_gpu_memory():
    logging.info("Sesión de Keras limpiada nuevamente.")
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


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

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


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

def extract_features(model, X, batch_size=32):
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_features = feature_extractor.predict(batch)
        features.append(batch_features)
    return np.concatenate(features)

def train_and_evaluate_ensemble(models_to_train, X_train, y_train, X_val, y_val, metadata_train, metadata_val, class_weights, class_names):
    all_train_features = []
    all_val_features = []
    
    for model_name in models_to_train:
        model = build_model(model_name, input_shape=(224, 224, 3), num_classes=y_train.shape[1])
        history, trained_model = train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights)
        
        train_features = extract_features(trained_model, X_train, batch_size=16)
        val_features = extract_features(trained_model, X_val, batch_size=16)
        
        all_train_features.append(train_features)
        all_val_features.append(val_features)
        
        trained_model.save(f'{model_name}_cnn_final.h5')
        logging.info(f"Modelo {model_name} guardado")
        
        clear_gpu_memory()
    
    X_train_cnn = np.hstack(all_train_features)
    X_val_cnn = np.hstack(all_val_features)
    
    meta_clf = MetaClassifier(classifier_type='auto', n_jobs=-1)
    
    # Corregimos esta línea para que coincida con la implementación de MetaClassifier
    meta_clf.fit(X_train_cnn, metadata_train, np.argmax(y_train, axis=1))
    
    y_pred = meta_clf.predict(X_val_cnn, metadata_val)
    y_pred_proba = meta_clf.predict_proba(X_val_cnn, metadata_val)
    
    y_true = np.argmax(y_val, axis=1)
    
    print("Reporte de clasificación para el Ensemble con MetaClassifier:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    plot_confusion_matrix(y_true, y_pred, class_names, "Ensemble")
    
    joblib.dump(meta_clf, 'ensemble_metaclassifier_final.joblib')
    logging.info("MetaClassifier del Ensemble guardado")
    
    class_metrics = get_metrics_per_class(y_true, y_pred, class_names)
    
    return class_metrics
def compare_models(results):
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    loss = [results[m]['loss'] for m in models]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(models, accuracy)
    plt.title('Comparación de Accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(models, loss)
    plt.title('Comparación de Loss')
    plt.ylabel('Validation Loss')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def get_metrics_per_class(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, index=class_names)
    return metrics

def print_metrics_table(results):
    metrics_df = pd.DataFrame(results).T
    metrics_df.columns = ['Accuracy', 'Loss']
    metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
    print("\nComparación de métricas totales:")
    print(metrics_df.to_string())

def print_metrics_per_class_table(all_class_metrics):
    print("\nComparación de métricas por clase:")
    for model, metrics in all_class_metrics.items():
        print(f"\n{model}:")
        print(metrics.to_string())

def main():
    # Cargar y preprocesar datos
    logging.info("Cargando datos...")
    X, y, metadata = load_data('IMGPRE', 'HAM10000_metadata.csv')
    X, y_encoded = preprocess_data(X, y)
    
    # Dividir datos
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
        X, y_encoded, metadata, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Calcular pesos de clase
    class_weights = dict(enumerate(y_encoded.sum(axis=0).max() / y_encoded.sum(axis=0)))
    
    class_names = np.unique(y)
    
    models_to_train = ['EfficientNetV2B0','Xception','DenseNet121', 'ResNet50', 'MobileNet', 'InceptionV3']
    
    logging.info("Iniciando entrenamiento del ensemble")
    class_metrics = train_and_evaluate_ensemble(
        models_to_train, X_train, y_train, X_val, y_val, 
        metadata_train, metadata_val, class_weights, class_names
    )
    
    logging.info("Entrenamiento del ensemble completado")
    
    # Imprimir métricas por clase
    print_metrics_per_class_table({'Ensemble': class_metrics})
    
    logging.info("Proceso completo finalizado")



if __name__ == "__main__":
    main()