import math
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================
#
# We keep ONE set of query heads fixed and only change how many
# KEY/VALUE heads exist. That is the single lever that separates
# MHA, GQA and MQA.
#
#   MHA : every query head has its own KV head   (n_kv = n_query)
#   GQA : query heads share KV heads in groups   (1 < n_kv < n_query)
#   MQA : all query heads share ONE KV head       (n_kv = 1)

d_model = 64
n_query_heads = 4
head_dim = d_model // n_query_heads     # 16

seq_len = 8
batch = 1


# ============================================================
# 1. KV-CACHE SIZE
# ============================================================
#
# Formula (elements stored during generation):
#
#   kv_cache = 2 * batch * n_kv_heads * seq_len * head_dim
#
#   the "2" is because we cache BOTH K and V.
#
# Only n_kv_heads changes between the variants, so the cache
# shrinks in direct proportion to the number of KV heads.

def kv_cache_size(n_kv_heads):
    return 2 * batch * n_kv_heads * seq_len * head_dim


# Number of KV heads for each variant
mha_kv_heads = n_query_heads     # 4
gqa_kv_heads = 2                 # 2 query heads per KV head
mqa_kv_heads = 1                 # all query heads share one KV head

mha_size = kv_cache_size(mha_kv_heads)
gqa_size = kv_cache_size(gqa_kv_heads)
mqa_size = kv_cache_size(mqa_kv_heads)


print("=" * 50)
print("KV-CACHE SIZE COMPARISON")
print("=" * 50)

print(f"\nd_model        : {d_model}")
print(f"query heads    : {n_query_heads}")
print(f"head_dim       : {head_dim}")
print(f"seq_len        : {seq_len}")

print("\n{:<6} {:>10} {:>14}".format("", "KV heads", "KV-cache size"))
print("-" * 32)
print("{:<6} {:>10} {:>14}".format("MHA", mha_kv_heads, mha_size))
print("{:<6} {:>10} {:>14}".format("GQA", gqa_kv_heads, gqa_size))
print("{:<6} {:>10} {:>14}".format("MQA", mqa_kv_heads, mqa_size))

print("\nMHA = {}".format(mha_size))
print("GQA = {}".format(gqa_size))
print("MQA = {}".format(mqa_size))

print("\nMHA > GQA > MQA:", mha_size > gqa_size > mqa_size)

print(
    "\nGQA cache is {:.0%} of MHA; MQA is {:.0%} of MHA.".format(
        gqa_size / mha_size,
        mqa_size / mha_size,
    )
)


# ============================================================
# 2. SOFTMAX BY HAND (for the GQA forward-pass check)
# ============================================================

# Formula:
# softmax(x_i) = exp(x_i) / sum_j exp(x_j)
def softmax_by_hand(x, dim=-1):
    max_values = x.max(dim=dim, keepdim=True).values
    shifted = x - max_values
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)


# ============================================================
# 3. TINY GQA FORWARD PASS (the bonus)
# ============================================================
#
# How GQA actually runs:
#   - project FEWER K/V heads than Q heads
#   - REPEAT each KV head so every query head in its group
#     gets a copy
#   - then it is ordinary multi-head attention
#
# group_size = how many query heads share one KV head.

torch.manual_seed(0)

group_size = n_query_heads // gqa_kv_heads     # 4 // 2 = 2

# Projections
W_q = nn.Linear(d_model, n_query_heads * head_dim, bias=False)   # 64 -> 64
W_k = nn.Linear(d_model, gqa_kv_heads * head_dim, bias=False)    # 64 -> 32
W_v = nn.Linear(d_model, gqa_kv_heads * head_dim, bias=False)    # 64 -> 32

# Small input: 3 tokens so the attention weights stay readable
T = 3
x = torch.randn(batch, T, d_model)


# --- project and split into heads ---

# Q: (B, T, 64) -> (B, T, 4, 16) -> (B, 4, T, 16)
Q = W_q(x).view(batch, T, n_query_heads, head_dim).transpose(1, 2)

# K, V: (B, T, 32) -> (B, T, 2, 16) -> (B, 2, T, 16)
K = W_k(x).view(batch, T, gqa_kv_heads, head_dim).transpose(1, 2)
V = W_v(x).view(batch, T, gqa_kv_heads, head_dim).transpose(1, 2)

print("\n" + "=" * 50)
print("GQA FORWARD PASS")
print("=" * 50)

print("\nQ shape (query heads):", tuple(Q.shape))
print("K shape (KV heads)   :", tuple(K.shape))
print("V shape (KV heads)   :", tuple(V.shape))

# --- repeat KV heads so each group's query heads get a copy ---
# (B, 2, T, 16) -> (B, 4, T, 16)
K = K.repeat_interleave(group_size, dim=1)
V = V.repeat_interleave(group_size, dim=1)

print("\nAfter repeating KV heads to match query heads:")
print("K shape:", tuple(K.shape))
print("V shape:", tuple(V.shape))

# --- scaled dot-product attention ---
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

weights_torch = torch.softmax(scores, dim=-1)
weights_hand = softmax_by_hand(scores, dim=-1)

output = weights_torch @ V
# combine heads: (B, 4, T, 16) -> (B, T, 64)
output = output.transpose(1, 2).contiguous().view(batch, T, d_model)

print("\nAttention weights shape:", tuple(weights_torch.shape))
print("Output shape:", tuple(output.shape))

print("\nSoftmax match:", torch.allclose(weights_hand, weights_torch, atol=1e-6))


# ============================================================
# 4. FINAL CHECKS
# ============================================================

print("\n" + "=" * 50)
print("FINAL CHECKS")
print("=" * 50)

print("MHA = 1024 :", mha_size == 1024)
print("GQA = 512  :", gqa_size == 512)
print("MQA = 256  :", mqa_size == 256)
print("MHA > GQA > MQA:", mha_size > gqa_size > mqa_size)
print("Softmax match:", torch.allclose(weights_hand, weights_torch, atol=1e-6))
print("GQA output shape preserved:", output.shape == x.shape)
