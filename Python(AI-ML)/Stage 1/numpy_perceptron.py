import numpy as np


def perceptron(x, w, b):
    z = np.dot(x, w) + b

    if z >= 0:
        output = 1
    else:
        output = 0

    return z, output


x = np.array([2, 4])
w = np.array([0.2, 0.5])
b = 0.7

z, output = perceptron(x, w, b)

print("z =", z)
print("output =", output)