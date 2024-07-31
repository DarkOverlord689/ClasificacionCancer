# ClasificacionCancer
Este proyecto consta de dos partes, la primera parte se encarga de realizar el entrenamiento del modelo, y la segunda parte se encarga de una interfaz gráfica dode se realiza la predicción 

## Entrenamiento 

# Detector de Melanoma Avanzado

## Descripción

Se presenta un detector de melanoma basado en aprendizaje profundo, diseñado para ayudar en la detección temprana de esta forma de cáncer de piel. El sistema utiliza modelos pre-entrenados para analizar imágenes de lesiones cutáneas y proporcionar una clasificación con un nivel de confianza.

## Estructura del proyecto

El proyecto se compone de los siguientes archivos principales:

* **main.py:** Punto de entrada de la aplicación, donde se gestiona la carga de datos, el preprocesamiento, el entrenamiento y la evaluación de los modelos.
* **meta_classifier.py:** Contiene la clase `MetaClassifier` utilizada para el clasificador de ensamblaje.
* **IMGPRE/:** Directorio que contiene las imágenes de las lesiones cutáneas.
* **HAM10000_metadata.csv:** Archivo CSV con los metadatos de las imágenes.

## Requisitos

Para ejecutar este proyecto, necesitarás las siguientes librerías:

* **os**
* **numpy**
* **tensorflow**
* **scikit-learn**
* **matplotlib**
* **seaborn**
* **pandas**
* **joblib**

###Instalación 
1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/DarkOverlord689/ClasificacionCancer.git
```
 2. ** Creació de ambiente virtual**
  ```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. ** Ejecuta el script principal**

```bash
python main.py
```

## GUI

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



###Instalación 
1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/DarkOverlord689/ClasificacionCancer.git
```
 2. ** Creació de ambiente virtual**
  ```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate  # En Windows
```
3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```
4. ** Ejecuta el script principal**

```bash
python main.py
```
