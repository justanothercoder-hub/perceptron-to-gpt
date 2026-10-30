import numpy as np
import matplotlib.pyplot as plt


# Formula:
# PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
def sinusoidal_positional_encoding(max_position, d_model):
    pe = np.zeros((max_position, d_model))

    for pos in range(max_position):
        for i in range(0, d_model, 2):

            denominator = 10000 ** (i / d_model)

            pe[pos, i] = np.sin(pos / denominator)

            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / denominator)

    return pe


# --------------------------------------------------
# Part 1: tiny inspectable example
# --------------------------------------------------

tiny_pe = sinusoidal_positional_encoding(
    max_position=4,
    d_model=4,
)

print("Tiny positional encoding:")
print(np.round(tiny_pe, 4))


print("\nPosition 0:")
print(np.round(tiny_pe[0], 4))

print("\nPosition 2:")
print(np.round(tiny_pe[2], 4))


# At position 0:
#
# sin(0) = 0
# cos(0) = 1
#
# Therefore every sin/cos pair should start:
#
# [0, 1]

expected_position_0 = np.array([
    0.0,
    1.0,
    0.0,
    1.0,
])

assert np.allclose(
    tiny_pe[0],
    expected_position_0,
)


# For d_model = 4:
#
# dims 0,1:
# denominator = 10000^(0/4) = 1
#
# dims 2,3:
# denominator = 10000^(2/4) = 100
#
# Therefore position 2 is:
#
# [
#   sin(2),
#   cos(2),
#   sin(2 / 100),
#   cos(2 / 100)
# ]

expected_position_2 = np.array([
    np.sin(2.0),
    np.cos(2.0),
    np.sin(0.02),
    np.cos(0.02),
])

print("\nExpected position 2:")
print(np.round(expected_position_2, 4))

assert np.allclose(
    tiny_pe[2],
    expected_position_2,
)

print("\nNUMERIC CHECK PASSED")


# --------------------------------------------------
# Part 2: build a larger encoding for visualization
# --------------------------------------------------

max_position = 100
d_model = 64

pe = sinusoidal_positional_encoding(
    max_position=max_position,
    d_model=d_model,
)


print("\nHeatmap array shape:")
print(pe.shape)

assert pe.shape == (100, 64)
assert np.all(pe <= 1.0)
assert np.all(pe >= -1.0)

print("SHAPE/RANGE CHECK PASSED")


# --------------------------------------------------
# Part 3: heatmap
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(
    pe,
    aspect="auto",
    origin="upper",
)

plt.xlabel("Embedding dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")

plt.colorbar(label="Encoding value")

plt.tight_layout()
plt.show()