import torch
from torch.utils.data import Dataset

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

print("\nDataset:", dataset)
print("Number of samples:", len(dataset))

print("\nSample 0:")
print(dataset[0])

print("\nSample 2:")
print(dataset[2])

print("X:")
print(X)

print("\ny:")
print(y)

print("\nNumber of samples:", len(X))
print("Number of features:", X.shape[1])

print("\nIndividual samples:")

for i in range(len(X)):
    print("Sample", i)
    print("Input :", X[i])
    print("Label :", y[i])

print("\nIterating through Dataset:")

for sample in dataset:
    print(sample)
