x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print( "xw0, xw1, xw2, b: ", xw0, xw1, xw2, b)

z = xw0 + xw1 + xw2 + b
print("z: ",z)

y = max(z, 0) # ReLU activation Function 
print("y: ",y)

# Backward Pass

# The derivative from the next layer
dvalue = 1.0

# Derivative of the ReLU and the Chain Rule
drelu_dz =  dvalue * (1. if z > 0 else 0.)

print("drelu_dz: ",drelu_dz)

# Partial Derivatives of the Multiplication, the Chain Rule
dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
print("drelu_dxw0: ", drelu_dxw0)

dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
print("drelu_dxw1: ", drelu_dxw1)

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2
print("drelu_dxw2: ", drelu_dxw2)

dsum_db = 1
drelu_db = drelu_dz * dsum_db
print("drelu_db: ", drelu_db)

# partial derivatives of the weight and inputs, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

print("drelu_dx0: ", drelu_dx0)
print("drelu_dx1: ", drelu_dx1)
print("drelu_dx2: ", drelu_dx2)
print("drelu_dw0: ", drelu_dw0)
print("drelu_dw1: ", drelu_dw1)
print("drelu_dw2: ", drelu_dw2)