import numpy as np

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

inputs = np.array([[1, 2, 3, 2.5],
                    [2., 5., -1., 2],
                    [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])


# dinputs with np.dot
dinputs = np.dot(dvalues, weights.T)
print("dinputs:\n",dinputs)

dweights = np.dot(inputs.T, dvalues)
print("dweights:\n",dweights)

dbiases = np.sum(dvalues, axis = 0, keepdims = True)
print("dbiases:\n", dbiases)