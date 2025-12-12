# Neural Network From Scratch (Pass/Fail Predictor) 🧠

This project implements an **Artificial Neural Network (ANN)** built purely in **Python** and **NumPy**, without using deep learning frameworks like TensorFlow, Keras, or PyTorch.

The goal of the model is to predict whether a student will **pass or fail** a subject based on two variables:
1. Number of lectures attended.
2. Dedicated study hours.

## 📋 Table of Contents
- [Features](#-features)
- [Network Architecture](#-network-architecture)
- [Prerequisites](#-prerequisites)
- [Installation and Usage](#-installation-and-usage)
- [Project Structure](#-project-structure)
- [Mathematical Foundations](#-mathematical-foundations)

## ✨ Features
- **Zero Frameworks:** All *Forward Propagation*, *Backpropagation*, and *Gradient Descent* logic is implemented manually.
- **Data Visualization:** Plots using `matplotlib` to understand the dataset distribution.
- **Interactive Prediction:** Script to test the model with user-inputted data.
- **Optimization:** Uses Sigmoid activation function and Binary Cross Entropy loss.

## 🏗 Network Architecture
The model is a Multilayer Perceptron (MLP) with the following topology:

* **Input Layer:** 2 neurons (Lectures, Study Hours).
* **Hidden Layer:** 3 neurons (with randomly initialized weights).
* **Output Layer:** 1 neuron (Probability of passing).
* **Activation Function:** Sigmoid (`1 / (1 + e^-z)`).
* **Bias:** Integrated into the matrix calculation (`+ 1` in the weighted sum).

## 🛠 Prerequisites
The project requires Python 3.x and the following libraries for matrix manipulation and plotting:
```bash
pip install numpy matplotlib
```

## 🚀 Installation 

1. Clone the repository:
```bash
git clone [https://github.com/tu-usuario/repo.git](https://github.com/tu-usuario/repo.git)
```

2. Visualize the dataset:
```bash
python Dataset.py
```
Or simply hit "run" in your code editor.

3. Train the Model: To execute the training algorithm and see how the Loss decreases iteration after iteration:
```bash
python Training.py
```

4. Test the Model (Prediction): To interact with the neural network using the trained weights:
```bash
python Testing.py
```

## 📂 Project Structure
- Dataset.py: Contains the training data dictionary and the logic to plot the points on a 2D plane.

- Training.py: The core of the project. Contains the neuron class, the loss function, and the main loop that executes Gradient Descent to adjust the weights.

- Testing.py: Uses the optimal weights obtained from training to make inferences on new data entered via console.

## 🧮 Mathematical Foundations
+ Forward Propagation: The dot product of inputs and weights is calculated and passed through the activation function: $$ z = (Inputs \cdot Weights) + 1 $$ $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

+ Loss Function: Binary Cross Entropy is used to measure the error: $$ Loss = -\frac{1}{N} \sum (y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})) $$

+ Backpropagation: Partial derivatives of the error with respect to each weight are calculated using the Chain Rule to update weights in the opposite direction of the gradient: $$ W_{new} = W_{current} - (learning_rate \cdot \frac{\partial Error}{\partial W}) $$
