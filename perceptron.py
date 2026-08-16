import numpy as np

x_input = [0.1, 0.5, 0.2]
weight = [0.4, 0.3, 0.6]
threshold = 0.5

def step(weighted_sum):
    if weighted_sum > threshold:
        return 1
    else:
        return 0

def perceptron():
    weighted_sum = 0

    for i in range(len(x_input)):
        weighted_sum += x_input[i] * weight[i]
        print(weighted_sum)

    return step(weighted_sum)

def sigmoid(weighted_sum):
    classfication  = 1.0 /(1.0+ (np.exp(-weighted_sum)))
    return classfication

def sigmoid_neuron():
    weighted_sum = 0

    for i in range(len(x_input)):
        weighted_sum += x_input[i] * weight[i]
        print(weighted_sum)
    weighted_sum -= threshold
    return sigmoid(weighted_sum)

