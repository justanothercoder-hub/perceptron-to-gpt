import numpy as np

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


def step(x):
    return (x >= 0).astype(int)


# ----- Forward pass -----

Z1 = X @ W1.T + b1

A1 = step(Z1)

Z2 = A1 @ W2 + b2

A2 = step(Z2)


print("Z1:")
print(Z1)

print()

print("A1:")
print(A1)

print()

print("Z2:")
print(Z2)

print()

print("Predictions:")
print(A2)

print()

print("Actual:")
print(y)