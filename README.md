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
```
## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone [https://github.com/tu-usuario/repo.git](https://github.com/tu-usuario/repo.git)
```

2. Visualizar el Dataset:
```bash
python Dataset.py
```
O simplemente darle a "ejecutar" en su editor de código

3.Entrenar el Modelo: Para ejecutar el algoritmo de entrenamiento y ver cómo disminuye la pérdida (Loss) iteración tras iteración:
```bash
python Training.py
```

4.Probar el Modelo (Predicción): Para interactuar con la red neuronal utilizando los pesos entrenados:
```bash
python Testing.py
```

##📂 Estructura del Proyecto
- Dataset.py: Contiene el diccionario de datos de entrenamiento y la lógica para graficar los puntos en un plano 2D.

- Training.py: El núcleo del proyecto. Contiene la clase neuron, la función de pérdida y el bucle principal que ejecuta el Descenso del Gradiente para ajustar los pesos.

- Testing.py: Utiliza los pesos óptimos obtenidos del entrenamiento para realizar inferencias sobre nuevos datos introducidos por consola.

##🧮 Fundamentos Matemáticos
+ Forward Propagation: Se calcula el producto punto de las entradas por los pesos y se pasa por la función de activación: $$ z = (Inputs \cdot Weights) + 1 $$ $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

+ Función de Costo (Loss): Se utiliza la Entropía Cruzada Binaria (Binary Cross Entropy) para medir el error: $$ Loss = -\frac{1}{N} \sum (y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})) $$

+ Backpropagation: Se calculan las derivadas parciales del error respecto a cada peso utilizando la Regla de la Cadena para actualizar los pesos en la dirección opuesta al gradiente: $$ W_{nuevo} = W_{actual} - (learning_rate \cdot \frac{\partial Error}{\partial W}) $$
