import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---- Build a synthetic dataset: noisy sine wave ----
torch.manual_seed(0)
np.random.seed(0)

n_points = 200
X_all = np.linspace(-3, 3, n_points).reshape(-1, 1)
y_all = np.sin(X_all) + np.random.normal(0, 0.2, size=X_all.shape)  # noisy sine

# shuffle then split 70% train / 30% test
indices = np.random.permutation(n_points)
split = int(0.7 * n_points)
train_idx, test_idx = indices[:split], indices[split:]

X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
y_test = torch.tensor(y_all[test_idx], dtype=torch.float32)


# ---- Two models: tiny (underfits) and huge (overfits) ----

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.net(x)


class HugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(model, epochs=2000, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = loss_fn(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

    return train_losses, test_losses


torch.manual_seed(0)
tiny_model = TinyModel()
tiny_train_losses, tiny_test_losses = train(tiny_model)

torch.manual_seed(0)
huge_model = HugeModel()
huge_train_losses, huge_test_losses = train(huge_model)

print("Tiny model  -> final train loss:", tiny_train_losses[-1], " final test loss:", tiny_test_losses[-1])
print("Huge model  -> final train loss:", huge_train_losses[-1], " final test loss:", huge_test_losses[-1])
print()
print("Underfitting looks like: BOTH train and test loss stay high (tiny model).")
print("Overfitting looks like: train loss keeps dropping but test loss stops improving")
print("or starts climbing back up (huge model) -- watch the plot.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(tiny_train_losses, label="Train loss")
axes[0].plot(tiny_test_losses, label="Test loss")
axes[0].set_title("Tiny model (underfitting)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(huge_train_losses, label="Train loss")
axes[1].plot(huge_test_losses, label="Test loss")
axes[1].set_title("Huge model (overfitting)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
