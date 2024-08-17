"""
    Este es el módulo principal que ejecuta el entrenamiento
    hace un llamado de las de los modulos auxiliares que contienen las funcionalidades 
    implementa un flujo de trabajo para entrenar y evaluar un ensemble de modelos de aprendizaje profundo     
    
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import print_metrics_per_class_table
from model_builder import load_data, preprocess_data, train_and_evaluate_ensemble

def main():
    # Cargar y preprocesar datos
    logging.info("Cargando datos...")
    # Carga los datos de imágenes y metadatos
    X, y, metadata = load_data('IMGPRE', 'HAM10000_metadata.csv')
    # Preprocesa los datos (posiblemente normalización, codificación one-hot, etc.)
    X, y_encoded = preprocess_data(X, y)
    
    # Dividir datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
        X, y_encoded, metadata, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Calcular pesos de clase para manejar el desbalance de clases
    class_weights = dict(enumerate(y_encoded.sum(axis=0).max() / y_encoded.sum(axis=0)))
    
    # Obtener los nombres únicos de las clases
    class_names = np.unique(y)
    
    # Lista de modelos a entrenar en el ensemble
    models_to_train = ['EfficientNetV2B0','Xception','DenseNet121', 'ResNet50', 'MobileNet', 'InceptionV3']
    
    logging.info("Iniciando entrenamiento del ensemble")
    # Entrenar y evaluar el ensemble de modelos
    class_metrics, overall_metrics, optimal_thresholds = train_and_evaluate_ensemble(
        models_to_train, X_train, y_train, X_val, y_val, 
        metadata_train, metadata_val, class_weights, class_names
    )
    
    logging.info("Entrenamiento del ensemble completado")
    
    # Imprimir métricas por clase
    print_metrics_per_class_table({'Ensemble': class_metrics})
    
    # Imprimir métricas generales
    print("\nMétricas generales:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Imprimir umbrales óptimos por clase
    print("\nUmbrales óptimos por clase:")
    for class_name, threshold in zip(class_names, optimal_thresholds):
        print(f"{class_name}: {threshold:.4f}")
    
    logging.info("Proceso completo finalizado")

if __name__ == "__main__":
    main()
