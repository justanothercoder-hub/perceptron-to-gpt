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
# HELPER FUNCTION — RoPE        <-- CHANGE 1
# ============================================================

# Formula:
# theta = position * frequency
#
# x' = x*cos(theta) - y*sin(theta)
# y' = x*sin(theta) + y*cos(theta)
#
# start_pos lets a one-token input during cached generation
# know its TRUE absolute position (e.g. token "D" is position 3,
# not position 0).

def apply_rope(tensor, start_pos=0):

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

    # CHANGED: positions now run from start_pos, not from 0.
    positions = torch.arange(
        start_pos,
        start_pos + seq_len,
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
# HELPER FUNCTION — CAUSAL ATTENTION      <-- CHANGE 2
# ============================================================

# Formula:
# Attention(Q,K,V)
# =
# softmax((QK^T / sqrt(head_dim)) + causal_mask) V
#
# With a cache, Q and K can have DIFFERENT lengths:
#   Q = the new token(s)        -> length query_len
#   K = cached + new keys       -> length key_len
# so a square mask no longer works. We build a rectangular
# mask from absolute positions instead.

def causal_attention(Q, K, V, past_len=0):

    # --------------------------------
    # 1. Scaled dot-product scores
    # --------------------------------

    attention_scores = (
        Q @ K.transpose(-2, -1)
    )

    attention_scores = (
        attention_scores
        / (head_dim ** 0.5)
    )

    # --------------------------------
    # 2. Causal mask (rectangular)
    # --------------------------------

    query_len = Q.shape[2]
    key_len = K.shape[2]

    # Query i sits at absolute position (past_len + i).
    query_positions = torch.arange(
        past_len,
        past_len + query_len,
        device=Q.device
    ).unsqueeze(1)

    # Keys occupy absolute positions 0 .. key_len - 1.
    key_positions = torch.arange(
        key_len,
        device=Q.device
    ).unsqueeze(0)

    # A query may attend to a key only if key_pos <= query_pos.
    allowed = (
        key_positions <= query_positions
    )

    attention_scores = attention_scores.masked_fill(
        ~allowed[None, None, :, :],
        float("-inf")
    )

    # --------------------------------
    # 3. Softmax
    # --------------------------------

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1
    )

    # --------------------------------
    # 4. Weighted sum of values
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
# MULTI-HEAD CAUSAL SELF-ATTENTION       <-- CHANGE 3
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

        # Count how many token positions K / V were
        # actually computed for (the whole point of the demo).
        self.k_tokens_computed = 0
        self.v_tokens_computed = 0


    # Formula:
    # counter = 0
    def reset_counter(self):
        self.k_tokens_computed = 0
        self.v_tokens_computed = 0


    # Formula:
    # K_total = concat(K_past, K_new)
    # V_total = concat(V_past, V_new)
    def forward(self, x, cache=None, use_cache=False):

        batch_size, seq_len, _ = x.shape

        # --------------------------------
        # 1. Q, K, V for the CURRENT input only
        # --------------------------------

        Q = self.query_projection(x)
        K_new = self.key_projection(x)
        V_new = self.value_projection(x)

        # We just computed K and V for (batch_size * seq_len)
        # token positions. Without a cache this is the full
        # sequence every step; with a cache it is only the new
        # token(s).
        self.k_tokens_computed += batch_size * seq_len
        self.v_tokens_computed += batch_size * seq_len

        # --------------------------------
        # 2. Split into heads
        # --------------------------------

        Q = split_heads(Q)
        K_new = split_heads(K_new)
        V_new = split_heads(V_new)

        # --------------------------------
        # 3. How many tokens are already cached?
        # --------------------------------

        if cache is not None:
            past_K, past_V = cache
            past_len = past_K.shape[2]
        else:
            past_K, past_V = None, None
            past_len = 0

        # --------------------------------
        # 4. RoPE at the TRUE absolute positions
        #
        # The new tokens sit at positions
        # [past_len, past_len + seq_len).
        #
        # We rotate ONLY the new K (K_new). The cached keys
        # (past_K) were already rotated at their own positions
        # when they were first computed, so we never re-rotate
        # them.
        # --------------------------------

        Q = apply_rope(Q, start_pos=past_len)
        K_new = apply_rope(K_new, start_pos=past_len)

        # --------------------------------
        # 5. Concatenate the cache  (the actual caching)
        # --------------------------------

        if cache is not None:
            K = torch.cat([past_K, K_new], dim=2)
            V = torch.cat([past_V, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        # --------------------------------
        # 6. Causal self-attention
        # --------------------------------

        attention_output = causal_attention(
            Q,
            K,
            V,
            past_len=past_len
        )

        # --------------------------------
        # 7. Combine heads + output projection
        # --------------------------------

        attention_output = combine_heads(attention_output)
        attention_output = self.output_projection(attention_output)

        # --------------------------------
        # 8. Return the updated cache (post-RoPE K, raw V)
        # --------------------------------

        if use_cache:
            new_cache = (K, V)
        else:
            new_cache = None

        return attention_output, new_cache


# ============================================================
# FEED-FORWARD NETWORK      (unchanged)
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
# TRANSFORMER BLOCK      <-- CHANGE 4
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
    def forward(self, x, cache=None, use_cache=False):

        # --------------------------------
        # 1. Pre-LN + Attention + Residual
        # --------------------------------

        attention_input = self.ln1(x)

        attention_output, new_cache = self.attention(
            attention_input,
            cache=cache,
            use_cache=use_cache
        )

        x = x + attention_output

        # --------------------------------
        # 2. Pre-LN + FFN + Residual
        # --------------------------------

        ffn_input = self.ln2(x)

        ffn_output = self.ffn(ffn_input)

        x = x + ffn_output

        return x, new_cache


# ============================================================
# TINY DECODER-ONLY GPT      <-- CHANGE 5
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
    # all counters = 0
    def reset_kv_counters(self):
        for block in self.blocks:
            block.attention.reset_counter()


    # Formula:
    # total = sum(count from each block)
    def get_kv_counts(self):
        total_k = sum(
            block.attention.k_tokens_computed
            for block in self.blocks
        )
        total_v = sum(
            block.attention.v_tokens_computed
            for block in self.blocks
        )
        return total_k, total_v


    # Formula:
    # logits =
    # LMHead(
    #   FinalLN(
    #     Block_N(...Block_2(Block_1(Embedding(tokens))))
    #   )
    # )
    def forward(self, token_ids, caches=None, use_cache=False):

        # --------------------------------
        # 1. Token embeddings
        # --------------------------------

        x = self.token_embedding(token_ids)
        # Shape: (B, T) -> (B, T, 32)

        # One cache per Transformer block
        if caches is None:
            caches = [None for _ in self.blocks]

        new_caches = []

        # --------------------------------
        # 2. Transformer blocks
        # --------------------------------

        for block, cache in zip(self.blocks, caches):
            x, new_cache = block(
                x,
                cache=cache,
                use_cache=use_cache
            )
            new_caches.append(new_cache)

        # --------------------------------
        # 3. Final LayerNorm
        # --------------------------------

        x = self.final_ln(x)

        # --------------------------------
        # 4. LM head
        # --------------------------------

        logits = self.lm_head(x)

        if use_cache:
            return logits, new_caches

        return logits


# ============================================================
# TINY GPT TEST      (leave it)
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
# TRAINING      (leave it)
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

    optimizer.zero_grad()

    logits = model(x)

    # Formula:
    # loss = CrossEntropy(predicted_logits, correct_targets)
    loss = criterion(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# LOSS PLOT      (leave it)
# ============================================================

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tiny GPT Training Loss")

plt.show()


# ============================================================
# SINGLE NEXT-TOKEN PREDICTION      (leave it)
# ============================================================

def predict_next_token(model, input_string):

    model.eval()

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

        last_token_logits = logits[:, -1, :]

        predicted_id = torch.argmax(
            last_token_logits,
            dim=-1
        ).item()

    predicted_token = id_to_token[predicted_id]

    return predicted_token


# ============================================================
# GENERATION WITHOUT KV-CACHE      <-- ADD
# ============================================================

# Formula:
# At every step the model re-encodes K and V for ALL tokens:
# K = X_all @ W_K
# V = X_all @ W_V

def generate_without_cache(model, prompt_ids, num_new_tokens):

    generated = prompt_ids.copy()
    model.eval()

    for _ in range(num_new_tokens):

        # Feed the ENTIRE sequence every step.
        input_tensor = torch.tensor(
            generated,
            dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)

        next_token_logits = logits[0, -1, :]

        next_id = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        generated.append(next_id)

    return generated


# ============================================================
# GENERATION WITH KV-CACHE      <-- ADD
# ============================================================

# Formula:
# K_cache = concat(K_cache, K_new)
# V_cache = concat(V_cache, V_new)
#
# The prompt is processed once; after that only the single
# newest token is fed in, and its K/V are appended to the cache.

def generate_with_cache(model, prompt_ids, num_new_tokens):

    generated = prompt_ids.copy()
    model.eval()

    caches = None

    # First call processes the full prompt.
    current_input = torch.tensor(
        generated,
        dtype=torch.long
    ).unsqueeze(0)

    for _ in range(num_new_tokens):

        with torch.no_grad():
            logits, caches = model(
                current_input,
                caches=caches,
                use_cache=True
            )

        next_token_logits = logits[0, -1, :]

        next_id = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        generated.append(next_id)

        # From now on, feed ONLY the new token.
        current_input = torch.tensor(
            [[next_id]],
            dtype=torch.long
        )

    return generated


# ============================================================
# KV-CACHE COMPARISON      <-- ADD
# ============================================================

prompt_tokens = ["A", "B", "C"]

prompt_ids = [
    token_to_id[token]
    for token in prompt_tokens
]

num_new_tokens = 5


# ----------------------------
# WITHOUT CACHE
# ----------------------------

model.reset_kv_counters()

uncached_ids = generate_without_cache(
    model,
    prompt_ids,
    num_new_tokens
)

uncached_k, uncached_v = model.get_kv_counts()


# ----------------------------
# WITH CACHE
# ----------------------------

model.reset_kv_counters()

cached_ids = generate_with_cache(
    model,
    prompt_ids,
    num_new_tokens
)

cached_k, cached_v = model.get_kv_counts()


# ----------------------------
# RESULTS
# ----------------------------

def ids_to_tokens(ids):
    return [id_to_token[i] for i in ids]


print("\n" + "=" * 50)
print("KV-CACHE COMPARISON")
print("=" * 50)

print("\nPrompt:", ids_to_tokens(prompt_ids))
print("New tokens to generate:", num_new_tokens)

print("\nWithout cache:", ids_to_tokens(uncached_ids))
print("With cache:   ", ids_to_tokens(cached_ids))

print("\nIdentical:", uncached_ids == cached_ids)

print("\nK/V token-positions computed:")
print(f"  without cache : {uncached_k}")
print(f"  with cache    : {cached_k}")
print(f"  work saved    : {uncached_k - cached_k} fewer positions")


# ============================================================
# INTERACTIVE PREDICTION
# (temporarily disabled for the KV-cache demo)
# ============================================================

# user_input = input(
#     "\nEnter a sequence using A-H: "
# )
#
# prediction = predict_next_token(
#     model,
#     user_input
# )
#
# print(
#     "Predicted next token:",
#     prediction
# )
