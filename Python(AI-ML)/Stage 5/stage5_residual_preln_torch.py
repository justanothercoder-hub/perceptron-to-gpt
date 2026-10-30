import torch
import torch.nn as nn


# ============================================================
# 1. RESIDUAL CONNECTION BY HAND
# ============================================================

# Formula:
# y = x + Sublayer(x)
def residual_by_hand(x, sublayer_output):
    return x + sublayer_output


# ============================================================
# 2. LAYERNORM BY HAND
# ============================================================

# Formula:
# mean = sum(x_i) / d
#
# variance = sum((x_i - mean)^2) / d
#
# LayerNorm(x) =
# (x - mean) / sqrt(variance + eps)
def layer_norm_by_hand(x, eps=1e-5):

    mean = x.mean(
        dim=-1,
        keepdim=True
    )

    variance = ((x - mean) ** 2).mean(
        dim=-1,
        keepdim=True
    )

    normalized = (
        (x - mean) /
        torch.sqrt(variance + eps)
    )

    return normalized


# ============================================================
# 3. SMALL INPUT
# ============================================================

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

print("========================================")
print("INPUT")
print("========================================")

print("\nx:")
print(x)

print("\nShape:")
print(x.shape)


# ============================================================
# 4. SIMPLE RESIDUAL EXAMPLE
# ============================================================

sublayer_output = torch.tensor([
    [0.5, -1.0],
    [1.0,  2.0],
])

residual_output = residual_by_hand(
    x,
    sublayer_output
)

print("\n========================================")
print("RESIDUAL CONNECTION")
print("========================================")

print("\nOriginal x:")
print(x)

print("\nSublayer output:")
print(sublayer_output)

print("\nx + Sublayer(x):")
print(residual_output)


# Expected:
#
# [1, 2] + [0.5, -1] = [1.5, 1]
#
# [3, 4] + [1, 2] = [4, 6]

expected_residual = torch.tensor([
    [1.5, 1.0],
    [4.0, 6.0],
])

print("\nExpected residual:")
print(expected_residual)

print("\nResidual calculation correct:")
print(
    torch.allclose(
        residual_output,
        expected_residual
    )
)


# ============================================================
# 5. LAYERNORM BY HAND
# ============================================================

ln_hand = layer_norm_by_hand(x)

print("\n========================================")
print("LAYERNORM BY HAND")
print("========================================")

print("\nInput:")
print(x)

print("\nLayerNorm by hand:")
print(ln_hand)


# ============================================================
# 6. PYTORCH LAYERNORM
# ============================================================

# elementwise_affine=False means:
# do not use learned gamma and beta
#
# This lets us compare directly with our hand implementation.

layer_norm_torch = nn.LayerNorm(
    normalized_shape=2,
    elementwise_affine=False
)

ln_torch = layer_norm_torch(x)

print("\nPyTorch LayerNorm:")
print(ln_torch)

print("\nLayerNorm by hand matches PyTorch:")
print(
    torch.allclose(
        ln_hand,
        ln_torch,
        atol=1e-6
    )
)


# ============================================================
# 7. SIMPLE DETERMINISTIC SUBLAYER
# ============================================================

# We use:
#
# F(x) = 2x
#
# so the numbers are easy to trace.

class DoubleSublayer(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            in_features=2,
            out_features=2,
            bias=False
        )

        # Make the layer calculate:
        #
        # [2 0]
        # [0 2]
        #
        # which means F(x) = 2x

        with torch.no_grad():

            self.linear.weight.copy_(
                torch.tensor([
                    [2.0, 0.0],
                    [0.0, 2.0],
                ])
            )

    # Formula:
    # F(x) = 2x
    def forward(self, x):
        return self.linear(x)


# ============================================================
# 8. VERIFY THE SIMPLE SUBLAYER
# ============================================================

sublayer = DoubleSublayer()

double_output = sublayer(x)

print("\n========================================")
print("SIMPLE SUBLAYER")
print("========================================")

print("\nx:")
print(x)

print("\nF(x) = 2x:")
print(double_output)

expected_double = 2 * x

print("\nSublayer really equals 2x:")
print(
    torch.allclose(
        double_output,
        expected_double
    )
)


# ============================================================
# 9. PRE-LN BY HAND
# ============================================================

# Formula:
# y = x + F(LN(x))
def pre_ln_by_hand(x, sublayer):

    normalized = layer_norm_by_hand(x)

    sublayer_output = sublayer(
        normalized
    )

    output = x + sublayer_output

    return output, normalized, sublayer_output


preln_output_hand, preln_normalized, preln_sublayer_output = (
    pre_ln_by_hand(
        x,
        DoubleSublayer()
    )
)

print("\n========================================")
print("PRE-LN BY HAND")
print("========================================")

print("\nOriginal x:")
print(x)

print("\nLN(x):")
print(preln_normalized)

print("\nF(LN(x)):")
print(preln_sublayer_output)

print("\nx + F(LN(x)):")
print(preln_output_hand)


# ============================================================
# 10. PRE-LN USING PYTORCH
# ============================================================

class PreLNBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln = nn.LayerNorm(
            normalized_shape=2,
            elementwise_affine=False
        )

        self.sublayer = DoubleSublayer()

    # Formula:
    # y = x + F(LN(x))
    def forward(self, x):

        return x + self.sublayer(
            self.ln(x)
        )


preln_torch_module = PreLNBlock()

preln_output_torch = preln_torch_module(x)

print("\nPre-LN PyTorch output:")
print(preln_output_torch)

print("\nPre-LN by hand matches PyTorch:")
print(
    torch.allclose(
        preln_output_hand,
        preln_output_torch,
        atol=1e-6
    )
)


# ============================================================
# 11. POST-LN BY HAND
# ============================================================

# Formula:
# y = LN(x + F(x))
def post_ln_by_hand(x, sublayer):

    sublayer_output = sublayer(x)

    residual = x + sublayer_output

    output = layer_norm_by_hand(
        residual
    )

    return output, sublayer_output, residual


postln_output_hand, postln_sublayer_output, postln_residual = (
    post_ln_by_hand(
        x,
        DoubleSublayer()
    )
)

print("\n========================================")
print("POST-LN BY HAND")
print("========================================")

print("\nOriginal x:")
print(x)

print("\nF(x):")
print(postln_sublayer_output)

print("\nx + F(x):")
print(postln_residual)

print("\nLN(x + F(x)):")
print(postln_output_hand)


# ============================================================
# 12. POST-LN USING PYTORCH
# ============================================================

class PostLNBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.sublayer = DoubleSublayer()

        self.ln = nn.LayerNorm(
            normalized_shape=2,
            elementwise_affine=False
        )

    # Formula:
    # y = LN(x + F(x))
    def forward(self, x):

        return self.ln(
            x + self.sublayer(x)
        )


postln_torch_module = PostLNBlock()

postln_output_torch = postln_torch_module(x)

print("\nPost-LN PyTorch output:")
print(postln_output_torch)

print("\nPost-LN by hand matches PyTorch:")
print(
    torch.allclose(
        postln_output_hand,
        postln_output_torch,
        atol=1e-6
    )
)


# ============================================================
# 13. COMPARE PRE-LN AND POST-LN
# ============================================================

print("\n========================================")
print("PRE-LN VS POST-LN")
print("========================================")

print("\nPre-LN:")
print("y = x + F(LN(x))")

print("\nPre-LN output:")
print(preln_output_torch)

print("\nPost-LN:")
print("y = LN(x + F(x))")

print("\nPost-LN output:")
print(postln_output_torch)


# ============================================================
# 14. SHAPE CHECK
# ============================================================

print("\n========================================")
print("SHAPE CHECK")
print("========================================")

print("\nInput shape:")
print(x.shape)

print("\nPre-LN output shape:")
print(preln_output_torch.shape)

print("\nPost-LN output shape:")
print(postln_output_torch.shape)

print(
    "\nPre-LN preserves shape:",
    x.shape == preln_output_torch.shape
)

print(
    "Post-LN preserves shape:",
    x.shape == postln_output_torch.shape
)


# ============================================================
# 15. FINAL CHECKS
# ============================================================

print("\n========================================")
print("FINAL CHECKS")
print("========================================")

print(
    "Residual calculation correct:",
    torch.allclose(
        residual_output,
        expected_residual
    )
)

print(
    "LayerNorm by hand matches PyTorch:",
    torch.allclose(
        ln_hand,
        ln_torch,
        atol=1e-6
    )
)

print(
    "Sublayer equals 2x:",
    torch.allclose(
        double_output,
        expected_double
    )
)

print(
    "Pre-LN by hand matches PyTorch:",
    torch.allclose(
        preln_output_hand,
        preln_output_torch,
        atol=1e-6
    )
)

print(
    "Post-LN by hand matches PyTorch:",
    torch.allclose(
        postln_output_hand,
        postln_output_torch,
        atol=1e-6
    )
)

print(
    "Pre-LN preserves shape:",
    x.shape == preln_output_torch.shape
)

print(
    "Post-LN preserves shape:",
    x.shape == postln_output_torch.shape
)