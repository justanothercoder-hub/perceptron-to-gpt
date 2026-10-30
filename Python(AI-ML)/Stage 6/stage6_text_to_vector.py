import torch
import torch.nn as nn


# ---------------------------------------------------------
# FIXED TOY VOCABULARY
# token -> integer ID
# ---------------------------------------------------------

vocab = {
    "the": 0,
    "cat": 1,
    "sat": 2,
    "on": 3,
    "mat": 4,
    "dog": 5,
}


# ---------------------------------------------------------
# Formula / mechanism:
#
# ids[i] = vocab[token_i]
#
# Tokenization here:
# text -> split into words -> vocabulary lookup
# ---------------------------------------------------------
def encode(text):
    tokens = text.split()

    ids = []

    for token in tokens:
        token_id = vocab[token]
        ids.append(token_id)

    return ids


# ---------------------------------------------------------
# INPUT TEXT
# ---------------------------------------------------------

text = "the cat sat on the mat"

print("=" * 60)
print("STAGE 1: RAW TEXT")
print("=" * 60)

print(text)


# ---------------------------------------------------------
# TOKENIZE / ENCODE
# ---------------------------------------------------------

ids = encode(text)

print("\n" + "=" * 60)
print("STAGE 2: TOKEN IDS")
print("=" * 60)

print("IDs:")
print(ids)

print("\nNumber of tokens:")
print(len(ids))


# Convert Python list -> PyTorch tensor

id_tensor = torch.tensor(ids, dtype=torch.long)

print("\nID tensor:")
print(id_tensor)

print("\nID tensor shape:")
print(id_tensor.shape)


# ---------------------------------------------------------
# EMBEDDING LAYER
#
# Formula / mechanism:
#
# embedding(token_id) = E[token_id]
#
# where:
#
# E has shape:
#
#     vocab_size x embedding_dim
#
# Here:
#
#     6 x 8
#
# ---------------------------------------------------------

vocab_size = len(vocab)
embedding_dim = 8

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)


# ---------------------------------------------------------
# IDs -> VECTORS
# ---------------------------------------------------------

vectors = embedding(id_tensor)

print("\n" + "=" * 60)
print("STAGE 3: EMBEDDING VECTORS")
print("=" * 60)

print(vectors)

print("\nEmbedding tensor shape:")
print(vectors.shape)


# ---------------------------------------------------------
# INSPECT THE EMBEDDING MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EMBEDDING MATRIX")
print("=" * 60)

print(embedding.weight)

print("\nEmbedding matrix shape:")
print(embedding.weight.shape)


# ---------------------------------------------------------
# VERIFY THAT nn.Embedding IS JUST A ROW LOOKUP
#
# By hand:
#
# vector_i = E[id_i]
# ---------------------------------------------------------

vectors_by_hand = embedding.weight[id_tensor]

print("\n" + "=" * 60)
print("BY-HAND ROW LOOKUP")
print("=" * 60)

print(vectors_by_hand)


# Verify PyTorch nn.Embedding matches direct row selection

print("\nDo nn.Embedding and manual row lookup match?")

print(
    torch.allclose(
        vectors,
        vectors_by_hand
    )
)


# ---------------------------------------------------------
# VERIFY REPEATED TOKENS GET THE SAME EMBEDDING
#
# Sentence:
#
# the cat sat on the mat
#
# position 0 = "the"
# position 4 = "the"
#
# Both IDs are 0.
# Therefore both select E[0].
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("REPEATED TOKEN CHECK")
print("=" * 60)

print("First 'the' vector:")
print(vectors[0])

print("\nSecond 'the' vector:")
print(vectors[4])

print("\nAre they identical?")

print(
    torch.allclose(
        vectors[0],
        vectors[4]
    )
)