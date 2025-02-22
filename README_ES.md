
# **PalmsCNN**

El repositorio **PalmsCNN** es una guía detallada que describe el proceso de entrenamiento de un modelo diseñado para segmentar y detectar las copas de tres especies de palmeras en los bosques amazónicos utilizando imágenes RGB capturadas por drones. Este trabajo forma parte de la investigación realizada por **Tagle et al.** (actualmente en revisión) en el estudio titulado **"Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms"** (Superando la brecha entre la investigación y la implementación mediante el mapeo basado en drones de palmeras amazónicas económicamente importantes).

El enfoque propuesto combina datos obtenidos mediante drones con una arquitectura avanzada que integra **ecoCNN** para la generación de datos, **DeepLabv3+** para la segmentación semántica y **DWT (Transformada Wavelet Discreta)** para el procesamiento de imágenes. Esta combinación permite una detección precisa y eficiente de las copas de las palmeras en imágenes aéreas.

#### **Especies Detectadas**
El modelo está entrenado para detectar tres especies de palmeras de importancia ecológica y económica en la Amazonía:
1. **Mauritia Flexuosa** (Clase 1)
2. **Euterpe Precatoria** (Clase 2)
3. **Oenocarpus Bataua** (Clase 3)

#### **Guía Paso a Paso**
El repositorio incluye una guía paso a paso que explica cómo:
1. Cargar un mosaico RGB obtenido por un dron (UAV).
2. Aplicar el modelo entrenado para detectar y segmentar las copas de las tres especies de palmeras.
3. Utilizar el enfoque **ecoCNN** junto con las arquitecturas **DeepLabv3+** y **DWT** para el procesamiento y análisis de imágenes.

#### **Tutorial de Entrenamiento**
Para aquellos interesados en replicar o entender el proceso de entrenamiento del modelo, el repositorio proporciona un tutorial detallado en el archivo **PalmsCNN_Tutorial**. Este tutorial cubre los pasos necesarios para generar datos, entrenar el modelo y validar los resultados.

#### **Contribución Científica**
Este trabajo busca cerrar la brecha entre la investigación científica y su aplicación práctica, demostrando cómo la tecnología de drones y el aprendizaje profundo pueden utilizarse para el mapeo y monitoreo de especies vegetales en ecosistemas complejos como la Amazonía. El código y la metodología presentados en este repositorio son una contribución valiosa para la comunidad científica y los profesionales en el campo de la ecología y la teledetección.

---

### **Tecnologías y Métodos Utilizados**
- **Datos de drones (UAV)**: Imágenes RGB de alta resolución.
- **ecoCNN**: Generación de datos para entrenamiento.
- **DeepLabv3+**: Arquitectura de segmentación semántica.
- **DWT (Transformada Wavelet Discreta)**: Procesamiento de imágenes para mejorar la detección.

---

Este repositorio es una herramienta invaluable para investigadores, ecólogos y profesionales interesados en el mapeo de vegetación y la aplicación de inteligencia artificial en estudios ambientales. ¡Explora el código y contribuye al avance de la ciencia! 🌿🤖

--- 

Si necesitas más detalles o ajustes, no dudes en decírmelo. 😊