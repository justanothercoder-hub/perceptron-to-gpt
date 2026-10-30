import numpy as np

# ---- Binary Cross-Entropy vs MSE, with concrete numbers ----

def bce_loss(y_pred, y_true, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# Walk through concrete cases first
print("---- BCE loss for individual predictions ----")
print("y=1, pred=0.99 -> BCE:", bce_loss(np.array([0.99]), np.array([1])))
print("y=1, pred=0.01 -> BCE:", bce_loss(np.array([0.01]), np.array([1])))
print("y=0, pred=0.01 -> BCE:", bce_loss(np.array([0.01]), np.array([0])))
print("y=0, pred=0.99 -> BCE:", bce_loss(np.array([0.99]), np.array([0])))
print()
print("Compare to MSE for the same 'very wrong' case (y=1, pred=0.01):")
print("MSE:", mse_loss(np.array([0.01]), np.array([1])))
print("BCE:", bce_loss(np.array([0.01]), np.array([1])))
print("-> BCE punishes a confidently wrong prediction MUCH harder than MSE does.")


# ---- Now compare the gradients on our actual XOR predictions ----
# Using our trained XOR predictions from Stage 3 as an example
A2 = np.array([0.011, 0.989, 0.989, 0.014])  # model's predicted outputs
y = np.array([0, 1, 1, 0])                    # actual XOR labels

n = len(y)

# MSE gradient w.r.t. A2 (what we've been using since Stage 2)
dA2_mse = (2 / n) * (A2 - y)

# BCE gradient w.r.t. A2 (before multiplying by sigmoid'(Z2))
# dL/dA2 for BCE = -(y/A2 - (1-y)/(1-A2)) / n
dA2_bce = -(y / A2 - (1 - y) / (1 - A2)) / n

print("\n---- Gradient comparison on near-converged XOR predictions ----")
print("A2 (predictions): ", A2)
print("y (actual):       ", y)
print("dA2 with MSE:     ", np.round(dA2_mse, 4))
print("dA2 with BCE:     ", np.round(dA2_bce, 4))

# ---- The key insight: for sigmoid + BCE, dZ2 simplifies beautifully ----
# When you multiply dA2_bce by sigmoid_derivative(Z2), the messy fraction cancels out
# and you're left with just: dZ2 = (A2 - y) / n
# No sigmoid-derivative multiplier choking the gradient when the prediction is very wrong.

dZ2_bce_simplified = (A2 - y) / n
print("\ndZ2 for BCE+sigmoid, simplified formula (A2 - y)/n:")
print(np.round(dZ2_bce_simplified, 4))
print("-> This is why BCE + sigmoid is the standard pairing for binary classification:")
print("   the gradient stays strong even when the model is confidently wrong.")
