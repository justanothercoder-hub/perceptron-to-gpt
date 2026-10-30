import math
import torch
import torch.nn as nn


# ============================================================
# 1. CAUSAL MASK
# ============================================================

# Formula:
# M[i, j] = True if j <= i
def causal_mask(seq_len):
    return torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )


# ============================================================
# 2. GELU BY HAND
# ============================================================

# Formula:
# GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
def gelu_by_hand(x):

    return 0.5 * x * (
        1.0
        + torch.erf(
            x / math.sqrt(2.0)
        )
    )


# ============================================================
# 3. MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(
            d_model,
            d_model
        )

        self.W_k = nn.Linear(
            d_model,
            d_model
        )

        self.W_v = nn.Linear(
            d_model,
            d_model
        )

        self.W_o = nn.Linear(
            d_model,
            d_model
        )


    # Formula:
    #
    # Attention(Q,K,V)
    # =
    # softmax(
    # mask(QK^T / sqrt(d_k))
    # )V
    #
    def forward(self, x):

        B = x.size(0)
        T = x.size(1)

        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)


        # ----------------------------------------------------
        # Split into heads
        #
        # [B, T, d_model]
        #
        # becomes
        #
        # [B, n_heads, T, d_k]
        # ----------------------------------------------------

        Q = Q.view(
            B,
            T,
            self.n_heads,
            self.d_k
        )

        K = K.view(
            B,
            T,
            self.n_heads,
            self.d_k
        )

        V = V.view(
            B,
            T,
            self.n_heads,
            self.d_k
        )


        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        # ----------------------------------------------------
        # QK^T
        # ----------------------------------------------------

        scores = Q @ K.transpose(
            -2,
            -1
        )


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scores = scores / math.sqrt(
            self.d_k
        )


        # ----------------------------------------------------
        # CAUSAL MASK
        # ----------------------------------------------------

        mask = causal_mask(T)

        # Change:
        #
        # [T, T]
        #
        # into:
        #
        # [1, 1, T, T]
        #
        # so the same mask broadcasts across
        # every batch and every attention head.

        mask = mask.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )


        # ----------------------------------------------------
        # SOFTMAX
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        # ----------------------------------------------------
        # Weighted sum of V
        # ----------------------------------------------------

        head_output = (
            attention_weights @ V
        )


        # ----------------------------------------------------
        # Put heads back together
        # ----------------------------------------------------

        head_output = head_output.transpose(
            1,
            2
        )

        head_output = (
            head_output
            .contiguous()
            .view(
                B,
                T,
                self.d_model
            )
        )


        # ----------------------------------------------------
        # Final projection
        # ----------------------------------------------------

        output = self.W_o(
            head_output
        )

        return output, attention_weights


# ============================================================
# 4. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.gelu = nn.GELU()

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )


    # Formula:
    #
    # FFN(x)
    # =
    # W2 GELU(W1x + b1) + b2
    #
    def forward(self, x):

        x = self.linear1(x)

        x = self.gelu(x)

        x = self.linear2(x)

        return x


# ============================================================
# 5. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            d_model
        )

        self.attention = MultiHeadAttention(
            d_model,
            n_heads
        )

        self.ln2 = nn.LayerNorm(
            d_model
        )

        self.ffn = FeedForward(
            d_model,
            d_ff
        )


    # Formula:
    #
    # x1 = x + MHA(LN1(x))
    #
    # y  = x1 + FFN(LN2(x1))
    #
    def forward(self, x):

        # ----------------------------------------
        # FIRST SUBLAYER
        #
        # LayerNorm
        #     ↓
        # Masked attention
        #     ↓
        # Residual add
        # ----------------------------------------

        normalized1 = self.ln1(x)

        attention_output, attention_weights = (
            self.attention(
                normalized1
            )
        )

        x = x + attention_output


        # ----------------------------------------
        # SECOND SUBLAYER
        #
        # LayerNorm
        #     ↓
        # FFN
        #     ↓
        # Residual add
        # ----------------------------------------

        normalized2 = self.ln2(x)

        ffn_output = self.ffn(
            normalized2
        )

        x = x + ffn_output


        return (
            x,
            attention_weights,
            normalized1,
            attention_output,
            normalized2,
            ffn_output
        )


# ============================================================
# 6. SMALL TEST INPUT
# ============================================================

B = 1
T = 3

d_model = 4
n_heads = 2
d_ff = 8


# Three tokens
# Four features per token

x = torch.tensor([
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
])


print("========================================")
print("INPUT")
print("========================================")

print("\nx:")
print(x)

print("\nInput shape:")
print(x.shape)


# ============================================================
# 7. CREATE TRANSFORMER BLOCK
# ============================================================

block = TransformerBlock(
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff
)


# ============================================================
# 8. FORWARD PASS
# ============================================================

(
    output,
    attention_weights,
    normalized1,
    attention_output,
    normalized2,
    ffn_output
) = block(x)


# ============================================================
# 9. INSPECT FIRST SUBLAYER
# ============================================================

print("\n========================================")
print("FIRST SUBLAYER: ATTENTION")
print("========================================")

print("\nAfter first LayerNorm:")
print(normalized1)

print("\nMasked attention output:")
print(attention_output)

print("\nAttention output shape:")
print(attention_output.shape)


# ============================================================
# 10. INSPECT CAUSAL ATTENTION
# ============================================================

print("\n========================================")
print("CAUSAL ATTENTION WEIGHTS")
print("========================================")

print("\nHead 1:")
print(attention_weights[0, 0])

print("\nHead 2:")
print(attention_weights[0, 1])


# Upper triangle should contain only zeros.

upper_triangle = torch.triu(
    attention_weights,
    diagonal=1
)

print("\nUpper triangle:")
print(upper_triangle)

causal_check = torch.all(
    upper_triangle == 0
)

print(
    "\nFuture attention is exactly zero:",
    causal_check
)


# ============================================================
# 11. INSPECT SECOND SUBLAYER
# ============================================================

print("\n========================================")
print("SECOND SUBLAYER: FFN")
print("========================================")

print("\nAfter second LayerNorm:")
print(normalized2)

print("\nFFN output:")
print(ffn_output)

print("\nFFN output shape:")
print(ffn_output.shape)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("FINAL TRANSFORMER BLOCK OUTPUT")
print("========================================")

print("\nOutput:")
print(output)

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)


# ============================================================
# 13. VERIFY GELU BUILT-IN
# ============================================================

gelu_builtin = nn.GELU()

gelu_test = torch.tensor([
    -1.0,
    0.0,
    1.0
])

gelu_hand = gelu_by_hand(
    gelu_test
)

gelu_torch = gelu_builtin(
    gelu_test
)

print("\n========================================")
print("GELU CHECK")
print("========================================")

print("\nGELU by hand:")
print(gelu_hand)

print("\nGELU PyTorch:")
print(gelu_torch)

print(
    "\nGELU by hand matches PyTorch:",
    torch.allclose(
        gelu_hand,
        gelu_torch,
        atol=1e-6
    )
)


# ============================================================
# 14. FINAL CHECKS
# ============================================================

print("\n========================================")
print("FINAL CHECKS")
print("========================================")

print(
    "Input and output shapes match:",
    x.shape == output.shape
)

print(
    "Attention output shape correct:",
    attention_output.shape == x.shape
)

print(
    "FFN output shape correct:",
    ffn_output.shape == x.shape
)

print(
    "Future attention exactly zero:",
    torch.all(
        torch.triu(
            attention_weights,
            diagonal=1
        ) == 0
    ).item()
)

print(
    "GELU by hand matches PyTorch:",
    torch.allclose(
        gelu_hand,
        gelu_torch,
        atol=1e-6
    )
)