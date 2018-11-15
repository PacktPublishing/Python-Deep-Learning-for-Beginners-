import numpy as np

# Perceptron
def predict_perceptron(inputs, weights):
    if np.dot(inputs, weights) > 0:
        return 1
    else:
        return 0

def predict_perceptron_proper(inputs, weights):

    def step_function(input):
        return 1 if input > 0 else 0

    def linear_model(inputs, weights):
        return np.dot(inputs, weights)

    return step_function(linear_model(inputs, weights))

def neuron(inputs, weights):

    def sigmoid_function(input):
        return 1 / (1 + np.exp(-1 * input))

    def linear_model(inputs, weights):
        return np.dot(inputs, weights)

    return sigmoid_function(linear_model(inputs, weights))

neural_network = neuron(neuron(inputs, weights1), weights2)
