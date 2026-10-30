import torch
import torch.nn as nn

model = nn.Linear(2, 1)

with torch.no_grad():
    model.weight.copy_(
        torch.tensor([
            [0.2, 0.5]
        ])
    )

    model.bias.copy_(
        torch.tensor([0.7])
    )

x = torch.tensor([[2.0, 4.0]])

z = model(x)

if z.item() >= 0:
    output = 1
else:
    output = 0

print("z =", z)
print("output =", output)