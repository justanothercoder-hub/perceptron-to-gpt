import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Same noisy-sine setup as stage4_overfit_demo.py
torch.manual_seed(0)
np.random.seed(0)

n_points = 200
X_all = np.linspace(-3, 3, n_points).reshape(-1, 1)
y_all = np.sin(X_all) + np.random.normal(0, 0.2, size=X_all.shape)

indices = np.random.permutation(n_points)
split = int(0.7 * n_points)
train_idx, test_idx = indices[:split], indices[split:]

X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
y_test = torch.tensor(y_all[test_idx], dtype=torch.float32)


class HugeModel(nn.Module):
    """Same oversized model as before, but now dropout is optional."""
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(model, epochs=2000, lr=0.01, weight_decay=0.0):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()  # IMPORTANT: dropout is only active in train mode
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = loss_fn(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()  # dropout turned off for evaluation
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

    return train_losses, test_losses


# 1) No regularization (baseline, same as overfit demo)
torch.manual_seed(0)
plain_model = HugeModel(dropout_p=0.0)
plain_train, plain_test = train(plain_model, weight_decay=0.0)

# 2) Dropout only
torch.manual_seed(0)
dropout_model = HugeModel(dropout_p=0.3)
dropout_train, dropout_test = train(dropout_model, weight_decay=0.0)

# 3) Weight decay (L2) only
torch.manual_seed(0)
l2_model = HugeModel(dropout_p=0.0)
l2_train, l2_test = train(l2_model, weight_decay=0.01)

print("No regularization -> train:", plain_train[-1], " test:", plain_test[-1])
print("Dropout(0.3)      -> train:", dropout_train[-1], " test:", dropout_test[-1])
print("Weight decay(0.01)-> train:", l2_train[-1], " test:", l2_test[-1])
print()
print("Look at the GAP between train and test loss in each case, not just the raw numbers.")
print("A smaller gap = less overfitting.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(plain_train, label="Train")
axes[0].plot(plain_test, label="Test")
axes[0].set_title("No regularization")
axes[0].legend()

axes[1].plot(dropout_train, label="Train")
axes[1].plot(dropout_test, label="Test")
axes[1].set_title("Dropout p=0.3")
axes[1].legend()

axes[2].plot(l2_train, label="Train")
axes[2].plot(l2_test, label="Test")
axes[2].set_title("Weight decay=0.01")
axes[2].legend()

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

plt.tight_layout()
plt.show()
