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

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(200000):
    model.zero_grad()

    output = model.forward(X)

    loss = loss_fn(output, y)

    loss.backward()

    optimizer.step()

    if epoch % 20000 == 0:
        print(epoch, loss.item())

    if epoch % 40000 == 0 and epoch != 0:
            print("\nOutput now:")
            print(output)




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


