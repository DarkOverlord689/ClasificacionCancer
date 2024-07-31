# ClasificacionCancer
ClasificacionMulticlase

## Detector de Melanoma Avanzado

### Descripción

Este proyecto presenta un detector de melanoma basado en aprendizaje profundo, diseñado para ayudar en la detección temprana de esta forma de cáncer de piel. El sistema utiliza modelos pre-entrenados para analizar imágenes de lesiones cutáneas y proporcionar una clasificación con un nivel de confianza.

### Estructura del proyecto

El proyecto se compone de los siguientes archivos principales:

* **main.py:** Punto de entrada de la aplicación, donde se crea la interfaz gráfica de usuario (GUI) y se gestionan las interacciones del usuario.
* **ui_components.py:** Define los componentes visuales de la interfaz, como botones, etiquetas y paneles.
* **detector.py:** Contiene la lógica principal de la detección, incluyendo la carga de modelos, el preprocesamiento de imágenes y la realización de predicciones.
* **load_and_predict.py:** Define funciones para cargar modelos pre-entrenados, preprocesar imágenes y realizar predicciones.
* **historial_pacientes.csv:** Archivo CSV donde se almacenan los resultados de las predicciones para cada paciente.

### Requisitos

Para ejecutar este proyecto, necesitarás las siguientes librerías:

* **PyQt6:** Para crear la interfaz gráfica de usuario.
* **OpenCV:** Para el procesamiento de imágenes.
* **NumPy:** Para operaciones numéricas.
* **TensorFlow:** Para la creación y ejecución de modelos de aprendizaje profundo.
* **Pillow (PIL):** Para manipulación de imágenes.


