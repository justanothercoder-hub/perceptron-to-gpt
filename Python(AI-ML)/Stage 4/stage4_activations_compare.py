import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Same XOR setup as Stage 3, but now we compare Sigmoid vs ReLU in the hidden layer.

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
    def __init__(self, hidden_activation, hidden_size=8):
        super().__init__()

        # Using 8 hidden neurons instead of 2 here on purpose: with only 2 ReLU
        # neurons, it's common for one (or both) to get stuck outputting 0 for every
        # input right from initialization -- the "dead ReLU" problem. With 8 neurons
        # there's enough redundancy that some always stay alive, so this comparison
        # is stable and actually shows the activation difference instead of a coin flip.
        self.hidden = nn.Linear(2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.hidden_activation = hidden_activation  # e.g. torch.sigmoid or torch.relu

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden_activation(x)

        x = self.output(x)
        x = torch.sigmoid(x)  # keep output layer as sigmoid — we still need a 0..1 prediction

        return x


def train(model, epochs=5000, lr=0.5):
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


torch.manual_seed(1)
sigmoid_model = XORModel(hidden_activation=torch.sigmoid)
sigmoid_losses = train(sigmoid_model)

torch.manual_seed(1)
relu_model = XORModel(hidden_activation=torch.relu)
relu_losses = train(relu_model)

print("Sigmoid model final loss:", sigmoid_losses[-1])
print("ReLU model final loss:   ", relu_losses[-1])

plt.plot(sigmoid_losses, label="Sigmoid hidden layer")
plt.plot(relu_losses, label="ReLU hidden layer")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Sigmoid vs ReLU on XOR (same starting weights)")
plt.legend()
plt.show()

# Note: with only 2 hidden neurons and a tiny dataset, the gap won't be huge —
# the real payoff of ReLU shows up in deeper networks. But watch the shape of the
# curve: ReLU often moves faster in the first few hundred epochs.
