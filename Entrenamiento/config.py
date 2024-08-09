# Configuraciones y constantes

# Rutas de datos
IMAGE_DIR = 'IMGPRE'
METADATA_PATH = 'HAMICIS.csv'

# Modelos a entrenar
MODELS_TO_TRAIN = ['EfficientNetV2B0', 'Xception', 'DenseNet121', 'ResNet50', 'MobileNet', 'InceptionV3']

# Parámetros de entrenamiento
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4

# Parámetros de validación
VALIDATION_SPLIT = 0.2

# Configuración de callbacks
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_FACTOR = 0.1
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-6

# Configuración de la imagen
IMAGE_SIZE = (224, 224)
def log():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
