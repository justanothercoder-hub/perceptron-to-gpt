import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================
#
# MoE replaces the single FFN in a block with several parallel
# FFN "experts" plus a lightweight ROUTER. For each token the
# router picks only top_k experts to actually run.
#
#   num_experts = total experts available  (capacity / params)
#   top_k       = experts actually run per token (compute)
#
# "Sparse activation": huge TOTAL parameter count, but only a
# small ACTIVE count per token.

d_model = 3
num_experts = 4
top_k = 2


# ============================================================
# 1. SOFTMAX BY HAND
# ============================================================

# Formula:
# softmax(x_i) = exp(x_i) / sum_j exp(x_j)
def softmax_by_hand(x, dim=-1):
    max_values = x.max(dim=dim, keepdim=True).values
    shifted = x - max_values
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)


# ============================================================
# 2. TOKENS
# ============================================================

# 3 tokens, each a 3-dim embedding.
# Deliberately different so they route differently.
tokens = torch.tensor([
    [1.0,  0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.5,  0.5,  1.0],
])

num_tokens = tokens.shape[0]

print("=" * 50)
print("TOKENS")
print("=" * 50)
print(tokens)
print("shape:", tuple(tokens.shape))


# ============================================================
# 3. ROUTER  (a single Linear: d_model -> num_experts)
# ============================================================

torch.manual_seed(0)

router = nn.Linear(d_model, num_experts)

# Formula:
# router_logits = x @ W^T + b
router_logits = router(tokens)

# By-hand version using the SAME weights
router_logits_hand = tokens @ router.weight.T + router.bias

print("\n" + "=" * 50)
print("ROUTER LOGITS  (score for each expert)")
print("=" * 50)
print(router_logits)
print("shape:", tuple(router_logits.shape), "= (tokens, experts)")

print("\nRouter by-hand == PyTorch Linear:",
      torch.allclose(router_logits_hand, router_logits, atol=1e-6))


# ============================================================
# 4. TOP-K EXPERT SELECTION
# ============================================================

# Formula:
# topk_values, topk_indices = top_k(router_logits)
topk_values, topk_indices = torch.topk(
    router_logits,
    k=top_k,
    dim=-1
)

print("\n" + "=" * 50)
print("TOP-K SELECTION")
print("=" * 50)
print("Chosen expert ids:")
print(topk_indices)
print("Their raw logits:")
print(topk_values)


# ============================================================
# 5. ROUTING WEIGHTS  (softmax over ONLY the chosen experts)
# ============================================================

# Formula:
# weight_j = softmax(topk_values)_j
routing_weights = torch.softmax(topk_values, dim=-1)
routing_weights_hand = softmax_by_hand(topk_values, dim=-1)

print("\n" + "=" * 50)
print("ROUTING WEIGHTS")
print("=" * 50)
print(routing_weights)
print("Row sums (should be 1):", routing_weights.sum(dim=-1))

print("\nSoftmax by-hand == torch.softmax:",
      torch.allclose(routing_weights_hand, routing_weights, atol=1e-6))


# ============================================================
# 6. THE EXPERTS  (one small FFN each)
# ============================================================

# Formula (per expert):
# expert(x) = W2 * GELU(W1 * x + b1) + b2
class Expert(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


experts = nn.ModuleList([Expert() for _ in range(num_experts)])


# ============================================================
# 7. RUN ONLY THE CHOSEN EXPERTS, WEIGHT, AND ADD
# ============================================================

# Formula:
# output_token = sum_j  weight_j * expert_{chosen_j}(token)
#
# Only top_k experts run per token -> sparse activation.

output = torch.zeros_like(tokens)

for t in range(num_tokens):
    for j in range(top_k):
        expert_id = topk_indices[t, j].item()
        weight = routing_weights[t, j]
        output[t] += weight * experts[expert_id](tokens[t])

print("\n" + "=" * 50)
print("FINAL MoE OUTPUT")
print("=" * 50)
print(output)
print("shape:", tuple(output.shape))


# ============================================================
# 8. WHO WENT WHERE
# ============================================================

print("\n" + "=" * 50)
print("ROUTING DECISIONS")
print("=" * 50)

for t in range(num_tokens):
    picks = []
    for j in range(top_k):
        e = topk_indices[t, j].item()
        w = routing_weights[t, j].item()
        picks.append(f"expert {e} ({w:.2f})")
    print(f"token {t}: " + "  +  ".join(picks))

# How many experts ever got used vs how many exist
used_experts = torch.unique(topk_indices)
print(f"\nExperts available : {num_experts}")
print(f"Experts actually run per token : {top_k}")
print(f"Distinct experts used across all tokens : {len(used_experts)}  -> {used_experts.tolist()}")


# ============================================================
# 9. FINAL CHECKS
# ============================================================

print("\n" + "=" * 50)
print("FINAL CHECKS")
print("=" * 50)

print("Router by-hand == PyTorch Linear:",
      torch.allclose(router_logits_hand, router_logits, atol=1e-6))

print("Softmax by-hand == torch.softmax:",
      torch.allclose(routing_weights_hand, routing_weights, atol=1e-6))

print("Each token runs exactly top_k experts:",
      topk_indices.shape[1] == top_k)

print("Routing weights per token sum to 1:",
      torch.allclose(routing_weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-6))
