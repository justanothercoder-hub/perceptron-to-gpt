import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- Part 1: visualize 3 LR schedules over 1000 epochs ----

epochs = 1000
base_lr = 0.1

dummy_param = torch.nn.Parameter(torch.zeros(1))
dummy_param.grad = torch.zeros_like(dummy_param)  # fake zero gradient so optimizer.step() is a harmless no-op

# Step decay: cut LR by half every 300 epochs
opt_step = torch.optim.SGD([dummy_param], lr=base_lr)
sched_step = torch.optim.lr_scheduler.StepLR(opt_step, step_size=300, gamma=0.5)

# Cosine annealing: smooth curve down to ~0 over all epochs
opt_cos = torch.optim.SGD([dummy_param], lr=base_lr)
sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cos, T_max=epochs)

# Warmup (linear, first 50 epochs) then cosine decay
opt_warm = torch.optim.SGD([dummy_param], lr=base_lr)
warmup_epochs = 50

def warmup_then_cosine(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # scales lr from ~0 up to base_lr
    # after warmup, cosine-decay the remaining epochs
    import math
    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))

sched_warm = torch.optim.lr_scheduler.LambdaLR(opt_warm, lr_lambda=warmup_then_cosine)

step_lrs, cos_lrs, warm_lrs = [], [], []

for epoch in range(epochs):
    step_lrs.append(opt_step.param_groups[0]["lr"])
    cos_lrs.append(opt_cos.param_groups[0]["lr"])
    warm_lrs.append(opt_warm.param_groups[0]["lr"])

    opt_step.step()
    opt_cos.step()
    opt_warm.step()
    sched_step.step()
    sched_cos.step()
    sched_warm.step()

plt.figure(figsize=(8, 5))
plt.plot(step_lrs, label="Step decay")
plt.plot(cos_lrs, label="Cosine annealing")
plt.plot(warm_lrs, label="Warmup + cosine")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("Learning rate schedules")
plt.legend()
plt.show()


# ---- Part 2: train XOR with constant LR vs cosine annealing ----

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


def train_constant_lr(epochs=10000, lr=0.5):
    torch.manual_seed(3)
    model = XORModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train_cosine_lr(epochs=10000, lr=0.5):
    torch.manual_seed(3)
    model = XORModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # <-- the only addition vs constant LR
        losses.append(loss.item())
    return losses


constant_losses = train_constant_lr()
cosine_losses = train_cosine_lr()

print("Constant LR final loss:", constant_losses[-1])
print("Cosine LR final loss:  ", cosine_losses[-1])

plt.plot(constant_losses, label="Constant LR = 0.5")
plt.plot(cosine_losses, label="Cosine annealed LR (starts at 0.5)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Constant vs Cosine-annealed learning rate on XOR")
plt.legend()
plt.show()
