import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))



    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        exp_sum = np.sum(exp_values, axis = 1, keepdims = True)
        exp_norm = exp_values / exp_sum

        self.output = exp_norm

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(classes = 3, samples = 100)

dense1 = Layer_Dense(2, 3)

dense1.forward(X)

activation1 = Activation_ReLU()

activation1.forward(dense1.output)

dense2 = Layer_Dense(3,3)

dense2.forward(activation1.output)

activation2 = Activation_softmax()

activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()

loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

predictions  = np.argmax(activation2.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis  = 1)
accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)