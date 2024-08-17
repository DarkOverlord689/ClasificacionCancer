
"""
    Este código define una clase MetaClassifier que implementa un clasificador que combina las características 
    extraídas de los modelos entreandos previamente  con metadatos adicionales. Las principales características son:

    Selecciona automáticamente el mejor clasificador entre varias opciones o usar uno específico.
    Utiliza un pipeline de scikit-learn para manejar el preprocesamiento (imputación y escalado) y la clasificación.
    Implementa búsqueda aleatoria de hiperparámetros para optimizar el rendimiento del clasificador.
    Maneja la liberación de memoria para mejorar la eficiencia en el uso de recursos.
    Proporciona métodos para entrenar el clasificador, realizar predicciones y obtener probabilidades de clase.

"""

#-----------------------librerias-----------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
import logging
import gc

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
import logging
import gc

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador que combina características de los modelos entrenados y metadatos.
    Puede seleccionar automáticamente el mejor clasificador o usar uno específico.
    """
    def __init__(self, classifier_type='auto', n_jobs=-1):
        """
        Inicializa el MetaClassifier.
        
        Parámetros:
        - classifier_type: Tipo de clasificador a usar ('auto' para selección automática)
        - n_jobs: Número de jobs para procesamiento paralelo
        """
        self.classifier_type = classifier_type
        self.n_jobs = n_jobs
        self.pipeline = None
    
    def fit(self, X_cnn, X_metadata, y):
        """
        Entrena el MetaClassifier.
        
        Parámetros:
        - X_cnn: Características extraídas de la CNN
        - X_metadata: Metadatos adicionales
        - y: Etiquetas
        
        Retorna:
        - self: Instancia entrenada del MetaClassifier
        """
        logging.info("Iniciando proceso de fit en OptimizedMetaClassifier")
        logging.info(f"Forma de X_cnn: {X_cnn.shape}, Forma de X_metadata: {X_metadata.shape}")
        
        X_combined = self._combine_features(X_cnn, X_metadata)
        logging.info(f"Características combinadas. Forma: {X_combined.shape}")
        
        if self.classifier_type == 'auto':
            logging.info("Iniciando selección automática de clasificador")
            self.pipeline = self._auto_select_classifier(X_combined, y)
        else:
            logging.info(f"Usando clasificador específico: {self.classifier_type}")
            self.pipeline = self._get_classifier_pipeline(self.classifier_type)
        
        logging.info("Iniciando entrenamiento del pipeline final")
        self.pipeline.fit(X_combined, y)
        
        logging.info("Entrenamiento completado")
        gc.collect()  # Liberación de memoria
        return self
    
    def predict(self, X_cnn, X_metadata):
        """
        Realiza predicciones usando el clasificador entrenado.
        
        Parámetros:
        - X_cnn: Características extraídas de la CNN
        - X_metadata: Metadatos adicionales
        
        Retorna:
        - predictions: Predicciones del clasificador
        """
        logging.info("Iniciando predicción")
        X_combined = self._combine_features(X_cnn, X_metadata)
        predictions = self.pipeline.predict(X_combined)
        gc.collect()  # Liberación de memoria
        return predictions
    
    def predict_proba(self, X_cnn, X_metadata):
        """
        Predice las probabilidades de clase usando el clasificador entrenado.
        
        Parámetros:
        - X_cnn: Características extraídas de la CNN
        - X_metadata: Metadatos adicionales
        
        Retorna:
        - probabilities: Probabilidades predichas para cada clase
        """
        logging.info("Iniciando predicción de probabilidades")
        X_combined = self._combine_features(X_cnn, X_metadata)
        probabilities = self.pipeline.predict_proba(X_combined)
        gc.collect()  # Liberación de memoria
        return probabilities
    
    def _combine_features(self, X_cnn, X_metadata):
        """
        Combina las características de CNN y los metadatos.
        
        Parámetros:
        - X_cnn: Características extraídas de la CNN
        - X_metadata: Metadatos adicionales
        
        Retorna:
        - X_combined: Características combinadas
        """
        logging.info("Combinando características")
        return np.hstack([X_cnn, X_metadata])
    
    def _get_classifier_pipeline(self, classifier_type):
        """
        Crea un pipeline de clasificación basado en el tipo de clasificador especificado.
        
        Parámetros:
        - classifier_type: Tipo de clasificador a usar
        
        Retorna:
        - pipeline: Pipeline de scikit-learn con preprocesamiento y clasificador
        """
        logging.info(f"Creando pipeline para {classifier_type}")
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        if classifier_type == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
        elif classifier_type == 'DecisionTree':
            clf = DecisionTreeClassifier(random_state=42)
        elif classifier_type == 'LogisticRegression':
            clf = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Clasificador no soportado: {classifier_type}")
        
        return Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('classifier', clf)
        ])
    
    def _auto_select_classifier(self, X, y):
        """
        Selecciona automáticamente el mejor clasificador mediante búsqueda aleatoria.
        
        Parámetros:
        - X: Características combinadas
        - y: Etiquetas
        
        Retorna:
        - best_pipeline: El mejor pipeline encontrado
        """
        logging.info("Iniciando selección automática de clasificador")
        classifiers = {
            'RandomForest': RandomForestClassifier(n_jobs=self.n_jobs, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        best_score = -np.inf
        best_pipeline = None
        
        for name, clf in classifiers.items():
            logging.info(f"Evaluando clasificador: {name}")
            pipeline = self._get_classifier_pipeline(name)
            
            param_distributions = self._get_param_distributions(name)
            
            random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=3, n_jobs=self.n_jobs, scoring='accuracy')
            random_search.fit(X, y)
            
            logging.info(f"Mejor puntuación para {name}: {random_search.best_score_}")
            
            if random_search.best_score_ > best_score:
                best_score = random_search.best_score_
                best_pipeline = random_search.best_estimator_
            
            gc.collect()  # Liberación de memoria después de cada iteración
        
        logging.info(f"Mejor clasificador seleccionado con puntuación: {best_score}")
        return best_pipeline
    
    def _get_param_distributions(self, classifier_name):
        """
        Obtiene la distribución de parámetros para la búsqueda aleatoria de hiperparámetros.
        
        Parámetros:
        - classifier_name: Nombre del clasificador
        
        Retorna:
        - param_distributions: Diccionario con las distribuciones de parámetros
        """
        logging.info(f"Obteniendo distribución de parámetros para {classifier_name}")
        if classifier_name == 'RandomForest':
            return {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        elif classifier_name == 'DecisionTree':
            return {
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif classifier_name == 'LogisticRegression':
            return {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear'],
                'classifier__max_iter': [100, 200, 300]
            }
        else:
            return {}
