import numpy as np

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1,1)
print(softmax_output)

print(np.eye(softmax_output.shape[0]))

print(softmax_output * np.eye(softmax_output.shape[0]))

print(np.diagflat(softmax_output))

print(np.dot(softmax_output, softmax_output.T))

print(softmax_output)
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))


# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        exp_sum = np.sum(exp_values, axis = 1, keepdims = True)
        exp_norm = exp_values / exp_sum
        self.output = exp_norm
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate output and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


    