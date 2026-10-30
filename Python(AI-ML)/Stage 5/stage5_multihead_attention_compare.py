import torch
import torch.nn as nn

d_model = 64
n_heads = 8

B = 2
T = 3

x = torch.arange(B * T * d_model, dtype=torch.float32)
x = x.view(B, T, d_model)


# PyTorch's built-in Multi-Head Attention
builtin_model = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=n_heads,
    batch_first=True
)


# Self-attention
output, attention_weights = builtin_model(
    x,
    x,
    x,
    need_weights=True,
    average_attn_weights=False
)


print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)