import numpy as np
import math 
import random 

from Dataset import dataset

NEURONS = 3
NUM_INPUTS = 2

def loss(y_actual,y_predicted):
    results = []
    for i in range(len(y_actual)):
        results.append(y_actual[i] * np.log(y_predicted[i]) + (1 - y_actual[i]) * np.log(1 - y_predicted[i]))
    bce = -np.mean(results)
    return bce
    
def sigmoid_derivative(x):
    return  x * (1 - x)

def binary_cross_entropy_derivative(y_true, output):
    return (output - y_true) / (output * (1 - output) + 1e-8)

def truncate(number: float, max_decimals: int) -> float:
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))

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
                                                                                                                             
#manual creation of neurons:
neuron1 = neuron(2)
neuron2 = neuron(2)
neuron3 = neuron(2)
finalneuron = neuron(3)
#These numbers are the result of iterating 
neuron1.modify_weights([2.648609788793334, 2.2783249121314624])
neuron2.modify_weights([-3.618640771781068, 6.488929824585565])
neuron3.modify_weights([8.141854671037414, -3.9216634880944903])
finalneuron.modify_weights([-3.125042405600723, 4.870923761807939, -3.7206370682726897])
step = 0.001

#Model Training
counter = 0
while counter < 2500:
    y_predicted = []
    #Testing the model with certain weights
    for keys in dataset.keys():
        key = list(keys)
        
        z1 = neuron1.calculate(key)
        z2 = neuron2.calculate(key)
        z3 = neuron3.calculate(key)
        output = finalneuron.calculate([z1,z2,z3])
        y_predicted.append(output)
        print('Real:',dataset[keys],'Output:',output)

        #Calculating the gradient after each iteration and the new weights: Formula is w-step*gradient where gradient is the derivative of the cost function with respect to W*
        derivative_constant = binary_cross_entropy_derivative(dataset[keys],output) * sigmoid_derivative(output)
        neuron1_weights = [
            float((neuron1.weights[0])-(step*derivative_constant*finalneuron.weights[0]*key[0])),      #w1
            float((neuron1.weights[1])-(step*derivative_constant*finalneuron.weights[0]*key[1]))]      #w2
        neuron2_weights = [
            float((neuron2.weights[0])-(step*derivative_constant*finalneuron.weights[1]*key[0])),      #w3
            float((neuron2.weights[1])-(step*derivative_constant*finalneuron.weights[1]*key[1]))]      #w4
        neuron3_weights = [
            float(neuron3.weights[0]-(step*derivative_constant*finalneuron.weights[2]*key[0])),        #w5
            float((neuron3.weights[1])-(step*derivative_constant*finalneuron.weights[2]*key[1]))]      #w6
        finalneuron_weights = [
            float((finalneuron.weights[0])-(step*derivative_constant*z1)),                             #w7
            float((finalneuron.weights[1])-(step*derivative_constant*z2)),                             #w8
            float((finalneuron.weights[2])-(step*derivative_constant*z3))]                             #w9

        #Modyfing the existing weights
        neuron1.modify_weights(neuron1_weights)
        neuron2.modify_weights(neuron2_weights)
        neuron3.modify_weights(neuron3_weights)
        finalneuron.modify_weights(finalneuron_weights)

    #Calculating the loss after the dataset
    loss_calculeted = loss(list(dataset.values()),y_predicted)
    print(loss_calculeted)
    counter += 1


print(neuron1.weights, neuron2.weights, neuron3.weights, finalneuron.weights)

#[2.648609788793334, 2.2783249121314624] [-3.618640771781068, 6.488929824585565] [8.141854671037414, -3.9216634880944903] [-3.125042405600723, 4.870923761807939, -3.7206370682726897]