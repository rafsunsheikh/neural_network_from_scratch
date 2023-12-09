import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer
class Dense_layer():

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward Pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient of inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU Activation
class Activation_ReLU():

    # Forward Pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output from the inputs
        self.output = np.maximum(0, inputs)

    # Backward Pass
    def backward(self, dvalues):
        # Since we need to modify the original values
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax():

    # Forward Pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probability
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)

        self.output = probabilities

    # Backward Pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            








