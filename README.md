# Red Neuronal Desde Cero (Pass/Fail Predictor) 🧠

Este proyecto implementa una **Red Neuronal Artificial (ANN)** construida puramente en **Python** y **NumPy**, sin utilizar frameworks de aprendizaje profundo como TensorFlow, Keras o PyTorch.

El objetivo del modelo es predecir si un estudiante **aprobará o reprobará** una materia basándose en dos variables:
1. Cantidad de clases asistidas.
2. Horas de estudio dedicadas.

## 📋 Tabla de Contenidos
- [Características](#-características)
- [Arquitectura de la Red](#-arquitectura-de-la-red)
- [Pre-requisitos](#-pre-requisitos)
- [Instalación y Uso](#-instalación-y-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Fundamentos Matemáticos](#-fundamentos-matemáticos)

## ✨ Características
- **Zero Frameworks:** Toda la lógica de *Forward Propagation*, *Backpropagation* y *Gradient Descent* está implementada manualmente.
- **Visualización de Datos:** Gráficos con `matplotlib` para entender la distribución del dataset.
- **Predicción Interactiva:** Script para probar el modelo con datos ingresados por el usuario.
- **Optimización:** Uso de función de activación Sigmoide y pérdida Binary Cross Entropy.

## 🏗 Arquitectura de la Red
El modelo es un Perceptrón Multicapa (MLP) con la siguiente topología:

* **Capa de Entrada:** 2 neuronas (Clases, Horas de estudio).
* **Capa Oculta:** 3 neuronas (con pesos inicializados aleatoriamente).
* **Capa de Salida:** 1 neurona (Probabilidad de aprobar).
* **Función de Activación:** Sigmoide (`1 / (1 + e^-z)`).
* **Bias:** Integrado en el cálculo matricial (`+ 1` en la suma ponderada).

## 🛠 Pre-requisitos
El proyecto requiere Python 3.x y las siguientes librerías para manejo de matrices y gráficos:

```bash
pip install numpy matplotlib
