import logging
import numpy as np
from sklearn.model_selection import train_test_split
from utils import print_metrics_per_class_table
from model_builder import load_data, preprocess_data, train_and_evaluate_ensemble




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
    class_metrics, overall_metrics, optimal_thresholds = train_and_evaluate_ensemble(
        models_to_train, X_train, y_train, X_val, y_val, 
        metadata_train, metadata_val, class_weights, class_names
    )
    
    logging.info("Entrenamiento del ensemble completado")
    
    # Imprimir métricas por clase
    print_metrics_per_class_table({'Ensemble': class_metrics})
    
    print("\nMétricas generales:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nUmbrales óptimos por clase:")
    for class_name, threshold in zip(class_names, optimal_thresholds):
        print(f"{class_name}: {threshold:.4f}")
    
    logging.info("Proceso completo finalizado")

if __name__ == "__main__":
    main()