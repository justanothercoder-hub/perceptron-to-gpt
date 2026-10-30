import numpy as np

# ---- LayerNorm by hand ----
# LayerNorm normalizes ACROSS FEATURES, separately for each sample (each row).
# This is different from what you might expect (normalizing a whole column/feature
# across the batch) -- that's BatchNorm instead.

np.random.seed(0)

# 4 samples, 3 features each -- deliberately on very different scales
x = np.array([
    [10.0, 0.1, 1000.0],
    [20.0, 0.2, 2000.0],
    [5.0,  0.05, 500.0],
    [15.0, 0.15, 1500.0],
])

print("Input x:")
print(x)

eps = 1e-5

# mean and std computed PER ROW (axis=1), not per column
mean = x.mean(axis=1, keepdims=True)
std = x.std(axis=1, keepdims=True)

print("\nPer-sample mean (one number per row):")
print(mean)
print("\nPer-sample std (one number per row):")
print(std)

normalized = (x - mean) / (std + eps)

print("\nNormalized x (each row now has mean ~0, std ~1):")
print(normalized)
print("\nVerify -- mean of each row after normalizing:", normalized.mean(axis=1))
print("Verify -- std of each row after normalizing: ", normalized.std(axis=1))

# ---- Now apply learnable gamma (scale) and beta (shift) ----
# In a real network these start at gamma=1, beta=0 and are LEARNED during training,
# so the network can undo the normalization if that turns out to be better.
n_features = x.shape[1]
gamma = np.ones(n_features)   # start at 1 (no scaling yet)
beta = np.zeros(n_features)   # start at 0 (no shifting yet)

output = gamma * normalized + beta

print("\nOutput after applying gamma=1, beta=0 (identical to normalized, for now):")
print(output)

# Let's manually set gamma/beta to something else to see what changes
gamma2 = np.array([2.0, 0.5, 1.0])
beta2 = np.array([1.0, 0.0, -1.0])

output2 = gamma2 * normalized + beta2
print("\nOutput after applying gamma=[2, 0.5, 1], beta=[1, 0, -1]:")
print(output2)
print("-> Notice each COLUMN got scaled/shifted differently, but the normalization")
print("   (per-row mean/std) already happened before that -- gamma/beta act per feature.")
