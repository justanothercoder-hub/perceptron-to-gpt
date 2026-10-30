import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

class XORDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

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

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

dataset = XORDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2)

losses = []

model.train() #tells we are training the model

for epoch in range(1000):

    epoch_loss = 0.0

    for iteration, (X_batch, y_batch) in enumerate(dataloader):

        optimizer.zero_grad()

        output = model(X_batch)

        loss = loss_fn(output, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(
                "Epoch:", epoch + 1,
                "Iteration:", iteration + 1,
                "Loss:", loss.item()
            )

    epoch_loss /= len(dataloader)

    losses.append(epoch_loss)

    if (epoch + 1) % 100 == 0:
        print(
            "Average Epoch Loss:",
            epoch_loss
        )

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()