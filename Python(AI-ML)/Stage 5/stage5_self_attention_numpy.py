import numpy as np


# Formula:
# Q = X @ W_Q
def make_queries(X, W_Q):
    return X @ W_Q


# Formula:
# K = X @ W_K
def make_keys(X, W_K):
    return X @ W_K


# Formula:
# V = X @ W_V
def make_values(X, W_V):
    return X @ W_V


# Formula:
# S = Q @ K^T
# S[i, j] = Q[i] dot K[j]
def attention_scores(Q, K):
    return Q @ K.T


# Formula:
# O_i = sum_j attention_weight[i, j] * V_j
def weighted_value_sum(weights, V):
    return weights @ V


# 3 tokens, embedding dimension = 4
X = np.array([
    [1, 0, 1, 0],   # The
    [0, 1, 1, 0],   # cat
    [1, 1, 0, 1]    # sat
], dtype=float)


# 4-dimensional input -> 2-dimensional Q/K/V
W_Q = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
], dtype=float)


W_K = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
], dtype=float)


W_V = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 1]
], dtype=float)


Q = make_queries(X, W_Q)
K = make_keys(X, W_K)
V = make_values(X, W_V)

scores = attention_scores(Q, K)


print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nScores = Q @ K.T:")
print(scores)


# We are NOT using softmax yet.
# These are manually chosen weights just to demonstrate
# how a weighted combination of Values works.

weights = np.array([
    [0.5, 0.3, 0.2],   # The
    [0.2, 0.5, 0.3],   # cat
    [0.1, 0.3, 0.6]    # sat
])
output = weighted_value_sum(weights, V)


print("\nManual weights (all tokens):")
print(weights)
print("\nRow sums (should all be 1):", weights.sum(axis=1))
print("\nNew representations for all tokens:")
print(output)