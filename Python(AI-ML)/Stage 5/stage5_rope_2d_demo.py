import math
import torch


# Formula:
# x' = x*cos(theta) - y*sin(theta)
# y' = x*sin(theta) + y*cos(theta)
def rotate_2d_by_hand(vector, theta):
    x = vector[0]
    y = vector[1]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta

    return torch.stack([
        x_rotated,
        y_rotated,
    ])


# Formula:
# R(theta) =
# [[cos(theta), -sin(theta)],
#  [sin(theta),  cos(theta)]]
#
# rotated = R(theta) @ vector
def rotate_2d_matrix(vector, theta):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rotation_matrix = torch.stack([
        torch.stack([
            cos_theta,
            -sin_theta,
        ]),
        torch.stack([
            sin_theta,
            cos_theta,
        ]),
    ])

    return rotation_matrix @ vector


# --------------------------------------------------
# Part 1:
# Verify a simple 90-degree rotation
# --------------------------------------------------

vector = torch.tensor([
    1.0,
    0.0,
])

theta = torch.tensor(
    math.pi / 2
)

manual = rotate_2d_by_hand(
    vector,
    theta,
)

matrix = rotate_2d_matrix(
    vector,
    theta,
)

expected = torch.tensor([
    0.0,
    1.0,
])


print("Original vector:")
print(vector)

print("\nAngle in degrees:")
print(
    theta.item()
    * 180
    / math.pi
)

print("\nRotated by hand:")
print(manual)

print("\nRotated with matrix:")
print(matrix)

print("\nExpected approximately:")
print(expected)


assert torch.allclose(
    manual,
    expected,
    atol=1e-6,
)

assert torch.allclose(
    manual,
    matrix,
    atol=1e-6,
)

print("\n90 DEGREE ROTATION CHECK PASSED")


# --------------------------------------------------
# Part 2:
# Position determines the rotation angle
# --------------------------------------------------

# Formula:
# theta_position = position * base_angle
def position_angle(position, base_angle):
    return position * base_angle


base_angle = torch.tensor(
    math.pi / 6
)

# pi/6 radians = 30 degrees


for position in range(4):

    theta_position = position_angle(
        position,
        base_angle,
    )

    rotated = rotate_2d_by_hand(
        vector,
        theta_position,
    )

    degrees = (
        theta_position.item()
        * 180
        / math.pi
    )

    rounded = (
        torch.round(rotated * 10000)
        / 10000
    )

    print(
        f"\nPosition {position}"
        f" -> angle {degrees:.1f} degrees"
    )

    print(
        "Rotated vector:",
        rounded,
    )


print("\nPOSITION ROTATION DEMO PASSED")