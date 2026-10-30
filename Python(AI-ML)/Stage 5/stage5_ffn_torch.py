import math
import torch
import torch.nn as nn


# ============================================================
# 1. GELU BY HAND
# ============================================================

# Formula:
# GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
def gelu_by_hand(x):
    return 0.5 * x * (
        1.0 + torch.erf(x / math.sqrt(2.0))
    )


# ============================================================
# 2. FFN BY HAND
# ============================================================

# Formula:
# FFN(x) = W2 * GELU(W1*x + b1) + b2
def ffn_by_hand(x, W1, b1, W2, b2):

    # First linear layer:
    # d_model -> d_ff
    hidden_pre_activation = x @ W1.T + b1

    # Non-linearity
    hidden = gelu_by_hand(
        hidden_pre_activation
    )

    # Second linear layer:
    # d_ff -> d_model
    output = hidden @ W2.T + b2

    return output, hidden_pre_activation, hidden


# ============================================================
# 3. SMALL INPUT
# ============================================================

# 2 tokens
# d_model = 2
x = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
])

print("========================================")
print("INPUT")
print("========================================")

print("\nx:")
print(x)

print("\nInput shape:")
print(x.shape)


# ============================================================
# 4. FIRST LINEAR LAYER WEIGHTS
# ============================================================

# d_model = 2
# d_ff = 4
#
# W1 shape = (4, 2)
#
# For a token [x1, x2], these rows create:
#
# hidden[0] = x1
# hidden[1] = x2
# hidden[2] = x1 + x2
# hidden[3] = x1 - x2

W1 = torch.tensor([
    [1.0,  0.0],
    [0.0,  1.0],
    [1.0,  1.0],
    [1.0, -1.0],
])

b1 = torch.tensor([
    0.0,
    0.0,
    0.0,
    0.0,
])


# ============================================================
# 5. SECOND LINEAR LAYER WEIGHTS
# ============================================================

# d_ff = 4
# d_model = 2
#
# W2 shape = (2, 4)

W2 = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
])

b2 = torch.tensor([
    0.0,
    0.0,
])


# ============================================================
# 6. RUN THE BY-HAND FFN
# ============================================================

output_hand, pre_gelu, post_gelu = ffn_by_hand(
    x,
    W1,
    b1,
    W2,
    b2
)

print("\n========================================")
print("BY-HAND FFN")
print("========================================")

print("\nAfter first linear layer:")
print(pre_gelu)

print("\nAfter GELU:")
print(post_gelu)

print("\nFinal by-hand FFN output:")
print(output_hand)


# ============================================================
# 7. CHECK THE FIRST TOKEN MANUALLY
# ============================================================

# First token:
# x = [1, 2]
#
# W1 creates:
#
# [1,
#  2,
#  1 + 2,
#  1 - 2]
#
# =
#
# [1, 2, 3, -1]

print("\nFirst token before GELU:")
print(pre_gelu[0])

expected_first_token = torch.tensor([
    1.0,
    2.0,
    3.0,
    -1.0
])

print("\nExpected first token:")
print(expected_first_token)

print("\nFirst linear calculation is correct:")
print(
    torch.allclose(
        pre_gelu[0],
        expected_first_token
    )
)


# ============================================================
# 8. VERIFY GELU AGAINST PYTORCH
# ============================================================

gelu_torch = nn.GELU()

gelu_output_torch = gelu_torch(
    pre_gelu
)

gelu_output_hand = gelu_by_hand(
    pre_gelu
)

print("\n========================================")
print("GELU CHECK")
print("========================================")

print("\nGELU by hand:")
print(gelu_output_hand)

print("\nPyTorch nn.GELU:")
print(gelu_output_torch)

print("\nGELU by hand matches nn.GELU:")
print(
    torch.allclose(
        gelu_output_hand,
        gelu_output_torch,
        atol=1e-6
    )
)


# ============================================================
# 9. PYTORCH FFN MODULE
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        # d_model -> d_ff
        self.linear1 = nn.Linear(
            in_features=2,
            out_features=4
        )

        self.gelu = nn.GELU()

        # d_ff -> d_model
        self.linear2 = nn.Linear(
            in_features=4,
            out_features=2
        )

    # Formula:
    # FFN(x) = W2 * GELU(W1*x + b1) + b2
    def forward(self, x):

        x = self.linear1(x)

        x = self.gelu(x)

        x = self.linear2(x)

        return x


# ============================================================
# 10. COPY OUR FIXED WEIGHTS INTO PYTORCH
# ============================================================

ffn = FeedForward()

with torch.no_grad():

    ffn.linear1.weight.copy_(W1)
    ffn.linear1.bias.copy_(b1)

    ffn.linear2.weight.copy_(W2)
    ffn.linear2.bias.copy_(b2)


# ============================================================
# 11. RUN PYTORCH FFN
# ============================================================

output_torch = ffn(x)

print("\n========================================")
print("PYTORCH FFN")
print("========================================")

print("\nPyTorch FFN output:")
print(output_torch)

print("\nBy-hand FFN output:")
print(output_hand)

print("\nBy-hand FFN matches PyTorch:")
print(
    torch.allclose(
        output_hand,
        output_torch,
        atol=1e-6
    )
)


# ============================================================
# 12. SHAPE CHECK
# ============================================================

print("\n========================================")
print("SHAPE CHECK")
print("========================================")

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output_torch.shape)

print("\nInput and output shapes match:")
print(
    x.shape == output_torch.shape
)


# ============================================================
# 13. FINAL CHECKS
# ============================================================

print("\n========================================")
print("FINAL CHECKS")
print("========================================")

print(
    "First linear calculation correct:",
    torch.allclose(
        pre_gelu[0],
        expected_first_token
    )
)

print(
    "GELU by hand matches nn.GELU:",
    torch.allclose(
        gelu_output_hand,
        gelu_output_torch,
        atol=1e-6
    )
)

print(
    "By-hand FFN matches PyTorch:",
    torch.allclose(
        output_hand,
        output_torch,
        atol=1e-6
    )
)

print(
    "Input and output shapes match:",
    x.shape == output_torch.shape
)