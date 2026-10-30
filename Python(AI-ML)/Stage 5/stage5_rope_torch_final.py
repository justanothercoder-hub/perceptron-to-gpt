import torch


# --------------------------------------------------
# Part 1: build RoPE frequencies / cache
# --------------------------------------------------

# Formula:
# theta_i = 1 / base^(2i / d)
#
# angle(position, i) = position * theta_i
#
# We store:
# cos(position * theta_i)
# sin(position * theta_i)
#
def build_rope_cache(max_position, d_model, base=10000.0):
    assert d_model % 2 == 0

    pair_count = d_model // 2

    pair_indices = torch.arange(
        pair_count,
        dtype=torch.float32,
    )

    frequencies = 1.0 / (
        base ** (2 * pair_indices / d_model)
    )

    positions = torch.arange(
        max_position,
        dtype=torch.float32,
    )

    angles = positions[:, None] * frequencies[None, :]

    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)

    return cos_cache, sin_cache, frequencies


# --------------------------------------------------
# Part 2: rotate one vector using RoPE
# --------------------------------------------------

# Formula for each pair (x, y):
#
# x' = x*cos(theta) - y*sin(theta)
# y' = x*sin(theta) + y*cos(theta)
#
def apply_rope(vector, position, cos_cache, sin_cache):
    assert vector.shape[-1] % 2 == 0

    rotated = torch.empty_like(vector)

    pair_count = vector.shape[-1] // 2

    for pair_index in range(pair_count):
        x_index = 2 * pair_index
        y_index = 2 * pair_index + 1

        x = vector[x_index]
        y = vector[y_index]

        cos_theta = cos_cache[position, pair_index]
        sin_theta = sin_cache[position, pair_index]

        rotated[x_index] = (
            x * cos_theta
            - y * sin_theta
        )

        rotated[y_index] = (
            x * sin_theta
            + y * cos_theta
        )

    return rotated


# --------------------------------------------------
# Part 3: inspect the frequencies
# --------------------------------------------------

d_model = 4
max_position = 10

cos_cache, sin_cache, frequencies = build_rope_cache(
    max_position=max_position,
    d_model=d_model,
)


print("Frequencies:")
print(frequencies)


# For d_model = 4:
#
# pair 0:
# theta_0 = 1 / 10000^(0/4)
#         = 1
#
# pair 1:
# theta_1 = 1 / 10000^(2/4)
#         = 1/100
#         = 0.01
#
expected_frequencies = torch.tensor([
    1.0,
    0.01,
])

assert torch.allclose(
    frequencies,
    expected_frequencies,
    atol=1e-6,
)

print("\nFREQUENCY CHECK PASSED")


# --------------------------------------------------
# Part 4: inspect one position by hand
# --------------------------------------------------

position = 2

print("\nPosition 2 angles:")
print(position * frequencies)

print("\nCos values at position 2:")
print(cos_cache[position])

print("\nSin values at position 2:")
print(sin_cache[position])


# Expected angles:
#
# pair 0:
# 2 * 1.0 = 2.0 radians
#
# pair 1:
# 2 * 0.01 = 0.02 radians
#
expected_angles = torch.tensor([
    2.0,
    0.02,
])

assert torch.allclose(
    position * frequencies,
    expected_angles,
    atol=1e-6,
)

print("\nANGLE CHECK PASSED")


# --------------------------------------------------
# Part 5: apply RoPE to a tiny vector
# --------------------------------------------------

q = torch.tensor([
    1.0,
    0.0,
    1.0,
    0.0,
])

print("\nOriginal Q:")
print(q)

q_rotated = apply_rope(
    q,
    position=2,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

print("\nQ rotated at position 2:")
print(q_rotated)


# Since each pair is [1, 0],
# rotation gives:
#
# pair 0:
# [cos(2), sin(2)]
#
# pair 1:
# [cos(0.02), sin(0.02)]
#
expected_q_rotated = torch.tensor([
    torch.cos(torch.tensor(2.0)),
    torch.sin(torch.tensor(2.0)),
    torch.cos(torch.tensor(0.02)),
    torch.sin(torch.tensor(0.02)),
])

assert torch.allclose(
    q_rotated,
    expected_q_rotated,
    atol=1e-6,
)

print("\nVECTOR ROTATION CHECK PASSED")


# --------------------------------------------------
# Part 6: prove relative-offset property
# --------------------------------------------------

q = torch.tensor([
    1.0,
    2.0,
    3.0,
    4.0,
])

k = torch.tensor([
    2.0,
    1.0,
    0.5,
    -1.0,
])


# Case A:
# Q at position 2
# K at position 5
#
# gap = 3
#
q_pos_2 = apply_rope(
    q,
    position=2,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

k_pos_5 = apply_rope(
    k,
    position=5,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

dot_a = torch.dot(
    q_pos_2,
    k_pos_5,
)


# Case B:
# Q at position 4
# K at position 7
#
# gap = 3
#
q_pos_4 = apply_rope(
    q,
    position=4,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

k_pos_7 = apply_rope(
    k,
    position=7,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

dot_b = torch.dot(
    q_pos_4,
    k_pos_7,
)


print("\nCase A:")
print("Q position = 2")
print("K position = 5")
print("Gap =", 5 - 2)
print("Q dot K =", dot_a.item())


print("\nCase B:")
print("Q position = 4")
print("K position = 7")
print("Gap =", 7 - 4)
print("Q dot K =", dot_b.item())


print("\nDifference:")
print(
    torch.abs(dot_a - dot_b).item()
)


assert torch.allclose(
    dot_a,
    dot_b,
    atol=1e-5,
)

print("\nRELATIVE OFFSET CHECK PASSED")


# --------------------------------------------------
# Part 7: show that a different gap changes the score
# --------------------------------------------------

k_pos_6 = apply_rope(
    k,
    position=6,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
)

dot_different_gap = torch.dot(
    q_pos_2,
    k_pos_6,
)

print("\nDifferent-gap case:")
print("Q position = 2")
print("K position = 6")
print("Gap =", 6 - 2)
print("Q dot K =", dot_different_gap.item())


print("\nSame-gap score:")
print(dot_a.item())

print("\nDifferent-gap score:")
print(dot_different_gap.item())


assert not torch.allclose(
    dot_a,
    dot_different_gap,
    atol=1e-5,
)

print("\nDIFFERENT GAP CHECK PASSED")

print("\nALL ROPE CHECKS PASSED")