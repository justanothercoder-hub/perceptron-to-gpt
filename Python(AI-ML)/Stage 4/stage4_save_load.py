import torch
import torch.nn as nn

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


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


loss_fn = nn.MSELoss()

# ---- Step 1: train a model for 50k epochs ----
torch.manual_seed(0)
model = XORModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(50000):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

print("Trained for 50,000 epochs. Loss:", loss.item())

# ---- Step 2: look at state_dict ----
print("\nstate_dict keys and shapes:")
for key, tensor in model.state_dict().items():
    print(f"  {key}: {tuple(tensor.shape)}")

# ---- Step 3: save just the weights ----
torch.save(model.state_dict(), "xor_weights.pth")
print("\nSaved weights to xor_weights.pth")

# ---- Step 4: create a brand-new model instance and load the weights into it ----
new_model = XORModel()  # freshly initialized -- random weights, untrained

with torch.no_grad():
    print("\nNew (untrained) model predictions BEFORE loading weights:")
    print(new_model(X))

new_model.load_state_dict(torch.load("xor_weights.pth"))

with torch.no_grad():
    print("\nNew model predictions AFTER loading weights:")
    print(new_model(X))
    print("\nOriginal model predictions (should match exactly):")
    print(model(X))


# ---- Step 5: save a full checkpoint (model + optimizer + epoch) so we can RESUME training ----
checkpoint = {
    "epoch": 50000,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}
torch.save(checkpoint, "xor_checkpoint.pth")
print("\nSaved full checkpoint to xor_checkpoint.pth")

# ---- Step 6: demonstrate resuming training from the checkpoint ----
resumed_model = XORModel()
resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)

loaded_checkpoint = torch.load("xor_checkpoint.pth")
resumed_model.load_state_dict(loaded_checkpoint["model_state_dict"])
resumed_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
start_epoch = loaded_checkpoint["epoch"]

print(f"\nResuming training from epoch {start_epoch}...")

for epoch in range(start_epoch, 100000):
    resumed_optimizer.zero_grad()
    loss = loss_fn(resumed_model(X), y)
    loss.backward()
    resumed_optimizer.step()

print("Finished training up to epoch 100,000. Final loss:", loss.item())

with torch.no_grad():
    print("\nFinal predictions after resumed training:")
    print(resumed_model(X))
    print("\nActual:")
    print(y)
