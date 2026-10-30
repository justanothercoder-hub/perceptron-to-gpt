import numpy as np


# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    0,
    1,
    1,
    0
])


# Hidden layer weights
W1 = np.array([
    [1, 1],
    [-1, -1]
])

# Hidden layer biases
b1 = np.array([
    -0.5,
    1.5
])


# Output layer weights
W2 = np.array([1, 1])

# Output layer bias
b2 = -1.5


# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Sigmoid derivative
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

x = np.array([-2, -1, 0, 1, 2])

print("Sigmoid:")
print(sigmoid(x))

print()

print("Sigmoid derivative:")
print(sigmoid_derivative(x))

# ----- Forward pass -----

Z1 = X @ W1.T + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

print()
print("A1:")
print(A1)

print()
print("A2:")
print(A2)

#loss
loss = np.mean((A2 - y) ** 2)

# ----- Backpropagation -----

n = len(y)

dA2 = (2 / n) * (A2 - y)

print()
print("dA2:")
print(dA2)


dZ2 = dA2 * sigmoid_derivative(Z2)

print()
print("dZ2:")
print(dZ2)

# Gradient of output layer weights
dW2 = A1.T @ dZ2

print()
print("dW2:")
print(dW2)

#Sum of all dZ2 since every Z uses same bias
db2 = np.sum(dZ2)
print("db2:")
print(db2)

print()
print("dW2:")
print(dW2)

print("Loss:")
print(loss)

