import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XORModelNoNorm(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden = nn.Linear(2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


class XORModelWithNorm(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden = nn.Linear(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)  # normalizes the hidden features per sample
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.norm(x)       # <-- LayerNorm applied BEFORE the activation
        x = torch.sigmoid(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


# Note on hidden_size=8 (rather than the 2 hidden units used elsewhere in this
# project): LayerNorm normalizes a vector to mean 0, std 1. If that vector only
# has 2 elements, standardizing forces them to always become +1 and -1 -- there's
# no room left to represent anything else, and training gets unreliable. With 8
# features LayerNorm has real signal to work with, which is the realistic case
# anyway (transformers normalize vectors with hundreds or thousands of dimensions).


def train(model, epochs=8000, lr=0.5):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


torch.manual_seed(0)
no_norm_model = XORModelNoNorm()
no_norm_losses = train(no_norm_model)

torch.manual_seed(0)
norm_model = XORModelWithNorm()
norm_losses = train(norm_model)

print("Without LayerNorm -> final loss:", no_norm_losses[-1])
print("With LayerNorm    -> final loss:", norm_losses[-1])
print()
print("On a network this tiny the difference may be small either way --")
print("the point of this exercise is the API: model.norm = nn.LayerNorm(hidden_size),")
print("applied right after the linear layer and before the activation. This exact")
print("pattern (Linear -> LayerNorm -> Activation) shows up inside every transformer block.")

plt.plot(no_norm_losses, label="No LayerNorm")
plt.plot(norm_losses, label="With LayerNorm")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("XOR training with vs without LayerNorm")
plt.legend()
plt.show()
