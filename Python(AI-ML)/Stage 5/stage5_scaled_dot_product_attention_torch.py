import math
import torch
import torch.nn.functional as F


# Formula:
# scores = Q @ K^T
def scaled_dot_product_attention(Q, K, V, mask=None):

    # Step 1: calculate raw attention scores
    scores = Q @ K.transpose(-2, -1)

    # Step 2: scale by sqrt(d_k)
    d_k = K.size(-1)
    scores = scores / math.sqrt(d_k)

    # Step 3: optional mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: softmax over the key dimension
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: weighted sum of Values
    output = attention_weights @ V

    return output, attention_weights


# Tiny example
Q = torch.tensor([
    [[2.0, 0.0],
     [1.0, 1.0],
     [1.0, 2.0]]
])

K = torch.tensor([
    [[2.0, 0.0],
     [1.0, 1.0],
     [1.0, 2.0]]
])

V = torch.tensor([
    [[2.0, 1.0],
     [1.0, 2.0],
     [1.0, 2.0]]
])


output, weights = scaled_dot_product_attention(Q, K, V)


print("Attention weights:")
print(weights)

print("\nOutput:")
print(output)

print("\nRow sums:")
print(weights.sum(dim=-1))


# Compare variance of raw and scaled dot-product scores//// why we divide by sqrt(d_k)??
torch.manual_seed(0)

d_k = 64
num_samples = 10000

Q_large = torch.randn(num_samples, d_k)
K_large = torch.randn(num_samples, d_k)

# Formula:
# score = Q · K
raw_scores = (Q_large * K_large).sum(dim=-1)

# Formula:
# scaled_score = (Q · K) / sqrt(d_k)
scaled_scores = raw_scores / math.sqrt(d_k)

print("\nVariance experiment:")
print("d_k:", d_k)
print("Raw score variance:", raw_scores.var().item())
print("Scaled score variance:", scaled_scores.var().item())
# end of this part ///

# Softmax saturation experiment

normal_scores = torch.tensor([2.0, 1.0, 0.0])
large_scores = torch.tensor([20.0, 10.0, 0.0])

normal_weights = F.softmax(normal_scores, dim=-1)
large_weights = F.softmax(large_scores, dim=-1)

print("\nSoftmax saturation experiment:")

print("\nNormal scores:")
print(normal_scores)

print("Softmax:")
print(normal_weights)

print("\nLarge scores:")
print(large_scores)

print("Softmax:")
print(large_weights)

# Softmax saturation experiment---END ///

# Final shape and attention-weight check

torch.manual_seed(0)

Q_test = torch.randn(1, 4, 8)
K_test = torch.randn(1, 4, 8)
V_test = torch.randn(1, 4, 8)

output_test, weights_test = scaled_dot_product_attention(
    Q_test,
    K_test,
    V_test
)

print("\nFinal attention test:")

print("Q shape:", Q_test.shape)
print("K shape:", K_test.shape)
print("V shape:", V_test.shape)

print("Attention weights shape:", weights_test.shape)
print("Output shape:", output_test.shape)

print("Row sums:")
print(weights_test.sum(dim=-1))