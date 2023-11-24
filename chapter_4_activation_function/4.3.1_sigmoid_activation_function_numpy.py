import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
exp_sum = np.sum(exp_values)
norm_exp = exp_values / exp_sum

print(norm_exp)