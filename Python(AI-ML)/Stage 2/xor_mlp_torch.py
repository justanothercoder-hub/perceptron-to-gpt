import torch
import torch.nn as nn

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)

        x = self.output(x)
        x = torch.sigmoid(x)

        return x


model = XORModel()

print(model)

loss_fn = nn.MSELoss()
learning_rate = 0.1

for epoch in range(10000):
    model.zero_grad()

    output = model.forward(X)

    loss = loss_fn(output, y)

    loss.backward()

    with torch.no_grad():
        model.hidden.weight -= learning_rate * model.hidden.weight.grad
        model.hidden.bias -= learning_rate * model.hidden.bias.grad

        model.output.weight -= learning_rate * model.output.weight.grad
        model.output.bias -= learning_rate * model.output.bias.grad

    if epoch % 1000 == 0:
        print(epoch, loss.item())



model.eval()

with torch.no_grad():
    final_output = model(X)

print("\nFinal predictions:")
print(final_output)

print("\nActual:")
print(y)

final_loss = loss_fn(final_output, y)

print("\nFinal loss:")
print(final_loss.item())

print("\nLoss:")
print(loss)
print("\nOutput:")
print(output)

# print("\nHidden weights:")
# print(model.hidden.weight)
#
# print("\nHidden bias:")
# print(model.hidden.bias)
#
# print("\nOutput weights:")
# print(model.output.weight)
#
# print("\nOutput bias:")
# print(model.output.bias)
#
# print("\nHidden weight gradients:")
# print(model.hidden.weight.grad)
#
# print("\nHidden bias gradients:")
# print(model.hidden.bias.grad)
#
# print("\nOutput weight gradients:")
# print(model.output.weight.grad)
#
# print("\nOutput bias gradients:")
# print(model.output.bias.grad)