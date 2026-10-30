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

W1 = np.array([
    [1, 1],
    [-1, -1]
])

b1 = np.array([
    -0.5,
    1.5
])

W2 = np.array([1, 1])

b2 = -1.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

n = len(y)
learning_rate = 0.1

for epoch in range(20000):
    Z1 = X @ W1.T + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    loss = np.mean((A2 - y) ** 2)

    dA2 = (2 / n) * (A2 - y)
    dZ2 = dA2 * sigmoid_derivative(Z2)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2)

    dA1 = np.outer(dZ2, W2)
    dZ1 = dA1 * sigmoid_derivative(Z1)

    dW1 = dZ1.T @ X
    db1 = np.sum(dZ1, axis=0)

    # update parameters
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if epoch % 2000 == 0:
        print(epoch, loss)


Z1 = X @ W1.T + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

print("\nFinal predictions:")
print(A2)

print("\nActual:")
print(y)