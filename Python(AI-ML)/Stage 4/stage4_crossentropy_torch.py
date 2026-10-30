import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def train(model, loss_fn, epochs=10000, lr=0.5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, output


torch.manual_seed(3)
mse_model = XORModel()
mse_losses, mse_final_output = train(mse_model, nn.MSELoss())

torch.manual_seed(3)
bce_model = XORModel()
bce_losses, bce_final_output = train(bce_model, nn.BCELoss())

print("MSE final loss:", mse_losses[-1])
print("BCE final loss:", bce_losses[-1])
print("\nMSE final predictions:", mse_final_output.detach().numpy().flatten())
print("BCE final predictions:", bce_final_output.detach().numpy().flatten())

# BCE loss values aren't directly comparable to MSE loss values (different scales),
# so don't compare the final numbers directly — instead look at the SHAPE of the curve
# and how fast each one drives predictions toward 0/1.

plt.plot(mse_losses, label="MSELoss")
plt.plot(bce_losses, label="BCELoss")
plt.xlabel("Epoch")
plt.ylabel("Loss (different scales — compare shape, not magnitude)")
plt.title("MSE vs BCE training on XOR (same starting weights)")
plt.legend()
plt.show()
