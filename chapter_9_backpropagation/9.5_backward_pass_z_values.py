import numpy as np

z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

drelu = np.zeros_like(z)
drelu[z > 0] = 1

print(drelu)

# The Chain Rule
drelu *= dvalues

print(drelu)