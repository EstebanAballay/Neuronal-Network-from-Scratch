import numpy as np
import math
import random
from Dataset import *  
import matplotlib.pyplot as plt

class neuron:
    def __init__(self,weight_number):
        #When creating a neuron it predifines a random weight 
        self.weights = [random.randint(-5,5) for i in range(weight_number)]

    def calculate(self,inputs):
        inarray = np.array(inputs)
        wearray = np.array(self.weights)
        z = (inarray @ wearray) + 1
        #z = np.clip(z, -500, 500)
        num = float(1 / (1 + math.exp(-z)))
        return num

    def modify_weights(self,new_weight):
        self.weights = new_weight

inputs = []
inputs.append(int(input("Ingrese la cantidad de materias que cursará: ")))
inputs.append(int(input("Ingrese la cantidad de horas que piensa destinar al proyecto: ")))
real_output = int(input("Escriba 0 si cree que fallara o 1 si cree que pasara: "))
dataset[tuple(inputs)] = real_output

#Creating the neurons and setting the correct weight
neuron1 = neuron(2)
neuron2 = neuron(2)
neuron3 = neuron(2)
finalneuron = neuron(3)
neuron1.modify_weights([7.885903004539852, -0.22518997615559916])
neuron2.modify_weights([-12.212332073604715, 10.55672908141972])
neuron3.modify_weights([14.907731214811664, -7.151805658861109])
finalneuron.modify_weights([-2.930507565121131, 5.072205968847784, -4.0999058105046355])

#Forward propagation,calculating the prediction
z1 = neuron1.calculate(inputs)
z2 = neuron2.calculate(inputs)
z3 = neuron3.calculate(inputs)
output = finalneuron.calculate([z1,z2,z3])

if round(output) == 1:
    print("La prediccion es que usted pasará")
else:
    print("Usted fallara este materia")

