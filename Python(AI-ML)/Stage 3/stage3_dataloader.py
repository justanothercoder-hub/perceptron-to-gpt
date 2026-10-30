import torch
from torch.utils.data import Dataset, DataLoader

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

dataset = XORDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2)

print("Dataset size:", len(dataset))
print("DataLoader:", dataloader)

for X_batch, y_batch in dataloader:
    print("\nX_batch:")
    print(X_batch)

    print("y_batch:")
    print(y_batch)

    print("X_batch shape:", X_batch.shape)
    print("y_batch shape:", y_batch.shape)