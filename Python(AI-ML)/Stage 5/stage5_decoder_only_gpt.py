import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# MODEL CONFIGURATION
# ============================================================

vocab_size = 8
context_window = 8

d_model = 32
num_heads = 4
head_dim = d_model // num_heads

d_ff = 128
num_blocks = 2


# ============================================================
# VOCABULARY
# ============================================================

token_to_id = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
}

id_to_token = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
}


# ============================================================
# NEXT-TOKEN TRAINING DATA
# ============================================================

# Formula:
# target[t] = sequence[t + 1]

sequence = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 0],
    dtype=torch.long
)

x = sequence[:-1]
y = sequence[1:]

# Add batch dimension
x = x.unsqueeze(0)
y = y.unsqueeze(0)


print("x:")
print(x)

print("\ny:")
print(y)

print("\nx shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# HELPER FUNCTION — SPLIT HEADS
# ============================================================

# Formula:
# (B, T, d_model)
# -> (B, T, num_heads, head_dim)
# -> (B, num_heads, T, head_dim)

def split_heads(tensor):

    batch_size, seq_len, _ = tensor.shape

    tensor = tensor.view(
        batch_size,
        seq_len,
        num_heads,
        head_dim
    )

    tensor = tensor.transpose(1, 2)

    return tensor


# ============================================================
# HELPER FUNCTION — RoPE
# ============================================================

# Formula:
# theta = position * frequency
#
# x' = x*cos(theta) - y*sin(theta)
# y' = x*sin(theta) + y*cos(theta)

def apply_rope(tensor):

    seq_len = tensor.shape[2]

    frequencies = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32,
                device=tensor.device
            )
            / head_dim
        )
    )

    positions = torch.arange(
        seq_len,
        dtype=torch.float32,
        device=tensor.device
    )

    angles = (
        positions[:, None]
        * frequencies[None, :]
    )

    cos_values = torch.cos(angles)
    sin_values = torch.sin(angles)

    # Broadcast across batch and heads
    cos_values = cos_values.unsqueeze(0).unsqueeze(0)
    sin_values = sin_values.unsqueeze(0).unsqueeze(0)

    # Dimension pairs:
    # (0,1), (2,3), (4,5), (6,7)

    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]

    rotated_even = (
        even * cos_values
        - odd * sin_values
    )

    rotated_odd = (
        even * sin_values
        + odd * cos_values
    )

    rotated = torch.empty_like(tensor)

    rotated[..., 0::2] = rotated_even
    rotated[..., 1::2] = rotated_odd

    return rotated


# ============================================================
# HELPER FUNCTION — CAUSAL ATTENTION
# ============================================================

# Formula:
# Attention(Q,K,V)
# =
# softmax((QK^T / sqrt(head_dim)) + causal_mask) V

def causal_attention(Q, K, V):

    # --------------------------------
    # Scaled dot-product scores
    # --------------------------------

    attention_scores = (
        Q @ K.transpose(-2, -1)
    )

    attention_scores = (
        attention_scores
        / (head_dim ** 0.5)
    )

    # --------------------------------
    # Causal mask
    # --------------------------------

    seq_len = Q.shape[2]

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=Q.device
        ),
        diagonal=1
    )

    attention_scores = attention_scores.masked_fill(
        causal_mask,
        float("-inf")
    )

    # --------------------------------
    # Softmax
    # --------------------------------

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1
    )

    # --------------------------------
    # Weighted sum of values
    # --------------------------------

    attention_output = (
        attention_weights @ V
    )

    return attention_output


# ============================================================
# HELPER FUNCTION — COMBINE HEADS
# ============================================================

# Formula:
# (B, num_heads, T, head_dim)
# -> (B, T, num_heads, head_dim)
# -> (B, T, d_model)

def combine_heads(tensor):

    batch_size, _, seq_len, _ = tensor.shape

    tensor = tensor.transpose(1, 2)

    tensor = tensor.contiguous().view(
        batch_size,
        seq_len,
        d_model
    )

    return tensor


# ============================================================
# MULTI-HEAD CAUSAL SELF-ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.query_projection = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


    # Formula:
    # MultiHeadAttention(x)
    # = W_O * Concat(
    #     Attention(
    #         RoPE(Q),
    #         RoPE(K),
    #         V
    #     )
    #   )

    def forward(self, x):

        # --------------------------------
        # 1. Q, K, V projections
        # --------------------------------

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # --------------------------------
        # 2. Split into heads
        # --------------------------------

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # --------------------------------
        # 3. Apply RoPE to Q and K
        # --------------------------------

        Q = apply_rope(Q)
        K = apply_rope(K)

        # --------------------------------
        # 4. Causal self-attention
        # --------------------------------

        attention_output = causal_attention(
            Q,
            K,
            V
        )

        # --------------------------------
        # 5. Combine heads
        # --------------------------------

        attention_output = combine_heads(
            attention_output
        )

        # --------------------------------
        # 6. Output projection
        # --------------------------------

        attention_output = self.output_projection(
            attention_output
        )

        return attention_output


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )


    # Formula:
    # FFN(x)
    # = W2 * GELU(W1*x + b1) + b2

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention()

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward()


    # Formula:
    # a = x + Attention(LN1(x))
    # y = a + FFN(LN2(a))

    def forward(self, x):

        # --------------------------------
        # 1. Pre-LN + Attention + Residual
        # --------------------------------

        attention_input = self.ln1(x)

        attention_output = self.attention(
            attention_input
        )

        x = x + attention_output


        # --------------------------------
        # 2. Pre-LN + FFN + Residual
        # --------------------------------

        ffn_input = self.ln2(x)

        ffn_output = self.ffn(
            ffn_input
        )

        x = x + ffn_output


        return x


# ============================================================
# TINY DECODER-ONLY GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # Token IDs -> 32-dimensional vectors
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Two Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(num_blocks)
            ]
        )

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(
            d_model
        )

        # Language-model head
        # 32 dimensions -> 8 vocabulary logits
        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )


    # Formula:
    # logits =
    # LMHead(
    #   FinalLN(
    #     Block_N(...Block_2(Block_1(Embedding(tokens))))
    #   )
    # )

    def forward(self, token_ids):

        # --------------------------------
        # 1. Token embeddings
        # --------------------------------

        x = self.token_embedding(
            token_ids
        )

        # Shape:
        # (B, T) -> (B, T, 32)


        # --------------------------------
        # 2. Transformer blocks
        # --------------------------------

        for block in self.blocks:

            x = block(x)


        # --------------------------------
        # 3. Final LayerNorm
        # --------------------------------

        x = self.final_ln(x)


        # --------------------------------
        # 4. LM head
        # --------------------------------

        logits = self.lm_head(x)

        return logits



# ============================================================
# TINY GPT TEST
# ============================================================

model = TinyGPT()

logits = model(x)

print("\n" + "=" * 50)
print("TINY GPT TEST")
print("=" * 50)

print("Input token shape:")
print(x.shape)

print("Final logits shape:")
print(logits.shape)

# ============================================================
# TRAINING
# ============================================================

model = TinyGPT()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)

epochs = 1000

loss_history = []


for epoch in range(epochs):

    # --------------------------------
    # 1. Clear old gradients
    # --------------------------------
    optimizer.zero_grad()


    # --------------------------------
    # 2. Forward pass
    # --------------------------------
    logits = model(x)


    # --------------------------------
    # 3. Calculate cross-entropy loss
    # --------------------------------

    # Formula:
    # loss = CrossEntropy(predicted_logits, correct_targets)

    loss = criterion(
        logits.view(-1, vocab_size),
        y.view(-1)
    )


    # --------------------------------
    # 4. Backpropagation
    # --------------------------------
    loss.backward()


    # --------------------------------
    # 5. AdamW parameter update
    # --------------------------------
    optimizer.step()


    # --------------------------------
    # 6. Store loss
    # --------------------------------
    loss_history.append(loss.item())


    # --------------------------------
    # 7. Print progress
    # --------------------------------
    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# LOSS PLOT
# ============================================================

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tiny GPT Training Loss")

plt.show()


# ============================================================
# SINGLE NEXT-TOKEN PREDICTION
# ============================================================

def predict_next_token(model, input_string):

    model.eval()

    # Example input:
    # "A B C D"

    tokens = input_string.upper().split()

    token_ids = [
        token_to_id[token]
        for token in tokens
    ]

    # Keep only the most recent context_window tokens
    token_ids = token_ids[-context_window:]

    input_tensor = torch.tensor(
        token_ids,
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(input_tensor)

        # Take logits from the LAST position only
        last_token_logits = logits[:, -1, :]

        predicted_id = torch.argmax(
            last_token_logits,
            dim=-1
        ).item()

    predicted_token = id_to_token[predicted_id]

    return predicted_token


# ============================================================
# INTERACTIVE PREDICTION
# ============================================================

# Wrapped in try/except so the script doesn't crash or hang when it
# is run without an interactive terminal (e.g. piped input, or an
# automated run). In PyCharm it still prompts normally.
try:

    user_input = input(
        "\nEnter a sequence using A-H: "
    )

    prediction = predict_next_token(
        model,
        user_input
    )

    print(
        "Predicted next token:",
        prediction
    )

except EOFError:

    print(
        "\nNo interactive input — skipping prediction."
    )