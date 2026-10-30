import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. ReLU
# =====================================================================

# Formula:  ReLU(x) = max(0, x)
#   -> if x is negative, output 0. if x is positive (or 0), output x unchanged.
def relu(x):
    return np.maximum(0, x)

# Derivative: ReLU'(x) = 1  if x > 0
#             ReLU'(x) = 0  if x < 0
#   -> the slope of "x" is 1. the slope of a flat 0 line is 0. There's a sharp
#      corner at x=0 where the derivative isn't technically defined; we just say 0.
def relu_derivative(x):
    return (x > 0).astype(float)


relu_test_points = np.array([-3, -1, 0, 1, 3])  # negative, negative, zero, positive, positive

print("---- 1. ReLU ----")
print("x:            ", relu_test_points)
print("relu(x):      ", relu(relu_test_points))
print("relu'(x):     ", relu_derivative(relu_test_points))
print("-> negative in, 0 out, 0 gradient. positive in, x out unchanged, gradient exactly 1.")
print("-> at x=0 the 'true' derivative is undefined; we just define it as 0 here, which is")
print("   what PyTorch does too (it doesn't matter in practice, x is almost never exactly 0).\n")


# =====================================================================
# 2. Tanh
# =====================================================================

# Formula:  tanh(x) = (e^x - e^-x) / (e^x + e^-x)
#   -> a ratio of two exponentials. As x grows large and positive, e^-x shrinks
#      to ~0, so this approaches e^x / e^x = 1. As x grows large and negative,
#      e^x shrinks to ~0, so it approaches -e^-x / e^-x = -1. That's why tanh is
#      squeezed between -1 and +1. NumPy already has this built in as np.tanh.
def tanh(x):
    return np.tanh(x)

# Derivative: tanh'(x) = 1 - tanh(x)^2
#   -> reuses tanh(x) itself, same "reuse the forward pass" trick as sigmoid.
#      At x=0, tanh(0)=0, so tanh'(0) = 1 - 0 = 1 (steepest point).
#      As |x| grows, tanh(x) approaches +-1, so tanh'(x) approaches 1 - 1 = 0.
def tanh_derivative(x):
    t = tanh(x)
    return 1 - t ** 2


tanh_test_points = np.array([-10, -3, -1, 0, 1, 3, 10])

print("---- 2. Tanh ----")
print("x:             ", tanh_test_points)
print("tanh(x):       ", np.round(tanh(tanh_test_points), 4))
print("tanh'(x):      ", np.round(tanh_derivative(tanh_test_points), 4))
print("-> output is always between -1 and +1 (unlike sigmoid's 0 to 1).")
print("-> look at tanh'(-10) and tanh'(10): both ~0.0000. Far from x=0 the gradient")
print("   vanishes almost completely, same failure mode as sigmoid, just centered at 0.\n")


# =====================================================================
# 3. GELU
# =====================================================================
# GELU = a smooth version of ReLU. Instead of a hard corner at x=0, it curves gently.
# You do NOT need to memorize this formula -- just know GELU(x) = x * (something between 0 and 1)
# that smoothly transitions like a soft ReLU. This is the tanh-based approximation PyTorch uses.

# Formula (exact):  GELU(x) = x * Phi(x)
#   where Phi(x) is the standard normal CDF -- "what fraction of a bell curve
#   lies to the left of x". So GELU multiplies x by a number between 0 and 1
#   that smoothly grows from ~0 (for very negative x) to ~1 (for very positive x).
#   That's the "smooth ReLU" idea: instead of a hard 0/1 gate like ReLU has,
#   the gate is a soft, continuously-changing probability.
#
# Formula (approximation used below, and inside PyTorch/most libraries):
#   GELU(x) ~= 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
#   -> this swaps the "exact" Phi(x) for a tanh-based curve that looks almost
#      identical but is much cheaper to compute. You do not need to memorize
#      this -- just recognize the shape: 0.5 * x * (1 + something between -1 and 1).
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# Derivative: no clean short formula (that's the tradeoff for GELU's smoothness).
# We estimate it numerically instead -- the definition of a derivative from
# calculus: slope = (f(x+h) - f(x-h)) / (2h) for a tiny step h. As h -> 0 this
# converges to the true slope of the curve at point x.
def gelu_derivative(x, h=1e-5):
    return (gelu(x + h) - gelu(x - h)) / (2 * h)


gelu_test_points = np.array([-3, -1, 0, 1, 3])

print("---- 3. GELU ----")
print("x:            ", gelu_test_points)
print("gelu(x):      ", np.round(gelu(gelu_test_points), 4))
print("gelu'(x):     ", np.round(gelu_derivative(gelu_test_points), 4))
print("-> compare gelu(-1) to relu(-1): ReLU gives exactly 0, GELU gives a small")
print("   negative number instead. That small 'leak' for negative inputs is the whole")
print("   difference -- it's ReLU with the sharp corner smoothed out.\n")


# =====================================================================
# 4. Sigmoid (reusing what we already know) + all 4 activations on ONE plot
# =====================================================================

# Formula:  sigmoid(x) = 1 / (1 + e^-x)
#   -> as x -> +infinity, e^-x -> 0, so this approaches 1 / 1 = 1.
#      as x -> -infinity, e^-x -> +infinity, so this approaches 1 / infinity = 0.
#      squeezes any real number into the range (0, 1). Same formula you've used
#      since Stage 2.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
#   -> reuses the forward-pass output s = sigmoid(x), no need to recompute
#      exponentials. Maximum value is at x=0 (s=0.5): 0.5 * 0.5 = 0.25.
#      Far from 0, s is near 0 or near 1, so s*(1-s) shrinks toward 0 either way --
#      this is the vanishing-gradient shape.
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


sigmoid_test_points = np.array([-3, -1, 0, 1, 3])

print("---- 4. Sigmoid ----")
print("x:               ", sigmoid_test_points)
print("sigmoid(x):      ", np.round(sigmoid(sigmoid_test_points), 4))
print("sigmoid'(x):     ", np.round(sigmoid_derivative(sigmoid_test_points), 4))
print("-> output squeezed between 0 and 1, same as we've used since Stage 2.\n")

x = np.linspace(-5, 5, 200)

plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, gelu(x), label="GELU")
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.title("Activation functions")
plt.xlabel("x")
plt.ylabel("activation(x)")
plt.legend()
plt.tight_layout()
plt.show()


# =====================================================================
# 5. Derivatives of all 4, on a SECOND plot -- this is where vanishing
#    gradient becomes visually obvious.
# =====================================================================

plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid_derivative(x), label="Sigmoid'")
plt.plot(x, tanh_derivative(x), label="Tanh'")
plt.plot(x, relu_derivative(x), label="ReLU'")
plt.plot(x, gelu_derivative(x), label="GELU'")
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.title("Activation derivatives (this is the vanishing-gradient picture)")
plt.xlabel("x")
plt.ylabel("derivative(x)")
plt.legend()
plt.tight_layout()
plt.show()

print("---- 5. What the derivative plot shows ----")
print("Sigmoid' and Tanh' both collapse toward 0 as |x| grows -- flat tails on both sides.")
print("ReLU' is a flat 0 for x<0 and a flat 1 for x>0 -- no shrinking on the positive side,")
print("no matter how large x gets. That's why deep networks lean on ReLU/GELU: gradients")
print("flowing backward through many layers don't get multiplied by tiny numbers over and over.")
