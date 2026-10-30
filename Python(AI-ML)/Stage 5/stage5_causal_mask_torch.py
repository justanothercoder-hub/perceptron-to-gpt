import math
import torch


# ============================================================
# 1. CAUSAL MASK
# ============================================================

# Formula:
# M[i, j] = 1 if j <= i else 0
def causal_mask_by_hand(seq_len):
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i, j] = True

    return mask


# Formula:
# M = lower_triangle(ones(seq_len, seq_len))
def causal_mask(seq_len):
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )


# ============================================================
# 2. SOFTMAX BY HAND
# ============================================================

# Formula:
# softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
def softmax_by_hand(x, dim=-1):
    # Subtract max for numerical stability.
    max_values = x.max(dim=dim, keepdim=True).values
    shifted = x - max_values

    exp_values = torch.exp(shifted)

    return exp_values / exp_values.sum(dim=dim, keepdim=True)


# ============================================================
# 3. SCALED DOT-PRODUCT ATTENTION
# ============================================================

# Formula:
# scores = QK^T / sqrt(d_k)
#
# masked_scores[i, j] = -inf if j > i
#
# weights = softmax(masked_scores)
#
# output = weights @ V
def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.shape[-1]

    # Step 1: QK^T
    scores = Q @ K.T

    # Step 2: scale by sqrt(d_k)
    scaled_scores = scores / math.sqrt(d_k)

    print("\nRaw scaled attention scores:")
    print(scaled_scores)

    # Step 3: apply causal mask
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        print("\nScores after applying causal mask:")
        print(scaled_scores)

    # Step 4: softmax
    weights_torch = torch.softmax(
        scaled_scores,
        dim=-1
    )

    weights_hand = softmax_by_hand(
        scaled_scores,
        dim=-1
    )

    print("\nAttention weights:")
    print(weights_torch)

    print("\nBy-hand softmax matches torch.softmax:")
    print(
        torch.allclose(
            weights_hand,
            weights_torch
        )
    )

    # Step 5: weighted sum of values
    output = weights_torch @ V

    return output, weights_torch


# ============================================================
# 4. VERIFY THE CAUSAL MASK ITSELF
# ============================================================

seq_len = 3

mask_hand = causal_mask_by_hand(seq_len)
mask_torch = causal_mask(seq_len)

print("========================================")
print("CAUSAL MASK")
print("========================================")

print("\nMask by hand:")
print(mask_hand)

print("\nMask using torch.tril:")
print(mask_torch)

print("\nDo the two masks match?")
print(
    torch.equal(
        mask_hand,
        mask_torch
    )
)


# ============================================================
# 5. SMALL, INSPECTABLE Q, K, V
# ============================================================

Q = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])

K = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])

V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

print("\n========================================")
print("INPUTS")
print("========================================")

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# ============================================================
# 6. ATTENTION WITH CAUSAL MASK
# ============================================================

mask = causal_mask(Q.shape[0])

print("\n========================================")
print("WITH CAUSAL MASK")
print("========================================")

output_masked, weights_masked = scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=mask
)

print("\nMasked attention output:")
print(output_masked)


# ============================================================
# 7. VERIFY FUTURE ATTENTION IS ZERO
# ============================================================

# diagonal=1 means:
# ignore the main diagonal and extract only values ABOVE it.
upper_triangle_masked = torch.triu(
    weights_masked,
    diagonal=1
)

print("\nUpper triangle of masked attention:")
print(upper_triangle_masked)

masked_check = torch.all(
    upper_triangle_masked == 0
)

print("\nAll future attention probabilities are exactly zero:")
print(masked_check)


# ============================================================
# 8. ATTENTION WITHOUT CAUSAL MASK
# ============================================================

print("\n========================================")
print("WITHOUT CAUSAL MASK")
print("========================================")

output_unmasked, weights_unmasked = scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=None
)

print("\nUnmasked attention output:")
print(output_unmasked)


# ============================================================
# 9. VERIFY FUTURE ATTENTION NOW EXISTS
# ============================================================

upper_triangle_unmasked = torch.triu(
    weights_unmasked,
    diagonal=1
)

print("\nUpper triangle of unmasked attention:")
print(upper_triangle_unmasked)

unmasked_check = torch.any(
    upper_triangle_unmasked != 0
)

print("\nWithout masking, future attention contains non-zero values:")
print(unmasked_check)


# ============================================================
# 10. FINAL EXERCISE CHECK
# ============================================================

print("\n========================================")
print("FINAL CHECKS")
print("========================================")

print(
    "Mask by-hand == torch.tril:",
    torch.equal(
        mask_hand,
        mask_torch
    )
)

print(
    "Masked future attention == 0:",
    torch.all(
        torch.triu(
            weights_masked,
            diagonal=1
        ) == 0
    ).item()
)

print(
    "Unmasked future attention has non-zero values:",
    torch.any(
        torch.triu(
            weights_unmasked,
            diagonal=1
        ) != 0
    ).item()
)