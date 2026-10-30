import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)


    # Formula:
    # Q = XW_q
    # K = XW_k
    # V = XW_v
    def forward(self, x):

        # creation of keys, queries, values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = Q.view(x.size(0), x.size(1), self.n_heads, self.d_k)
        K = K.view(x.size(0), x.size(1), self.n_heads, self.d_k)
        V = V.view(x.size(0), x.size(1), self.n_heads, self.d_k)

        # Move the head dimension before the sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.d_k)

        # Convert scores into attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to V
        head_output = attention_weights @ V

        # Move heads next to each token
        head_output = head_output.transpose(1, 2)

        # Concatenate all heads
        head_output = head_output.contiguous().view(
            x.size(0),
            x.size(1),
            self.d_model
        )

        # Final output projection
        output = self.W_o(head_output)

        return Q, K, V, scores, attention_weights,  head_output, output


d_model = 64
n_heads = 8

model = MultiHeadAttention(d_model, n_heads)

B = 2
T = 3

x = torch.arange(B * T * d_model, dtype=torch.float32)
x = x.view(B, T, d_model)

Q, K, V, scores, attention_weights, head_output, output = model(x)

print("Input shape:", x.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("d_k:", model.d_k)
print("Scores shape:", scores.shape)
print("Attention weights shape:", attention_weights.shape)
print("Attention weights:", attention_weights[0, 0])
print("Head output shape:", head_output.shape)
print("Final output shape:", output.shape)
print("\nHead 1 attention weights:")
print(attention_weights[0, 0])

print("\nHead 2 attention weights:")
print(attention_weights[0, 1])