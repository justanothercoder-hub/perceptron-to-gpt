def perceptron(x, w, b):
    z = 0

    for i in range(len(x)):
        z = z + x[i] * w[i]

    z = z + b

    if z >= 0:
        output = 1
    else:
        output = 0

    return output


x = [2, 4]
w = [0.2, 0.5]
b = 0.7

output = perceptron(x, w, b)

print(output)