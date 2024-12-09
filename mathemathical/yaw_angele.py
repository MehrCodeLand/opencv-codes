import numpy as np

def rotation_matrix_to_euler_angles(R):
    """
    Calculate Euler angles (yaw, pitch, roll) from a rotation matrix.
    :param R: 3x3 rotation matrix
    :return: tuple of yaw, pitch, and roll angles in radians
    """
    assert R.shape == (3, 3)

    # Calculate pitch (θ)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return yaw, pitch, roll

# Example usage
R = np.array([
    [0.36, -0.48, 0.8],
    [0.8, 0.6, 0],
    [-0.48, 0.64, 0.6]
])
yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

print(f"Yaw: {np.degrees(yaw):.2f} degrees")
print(f"Pitch: {np.degrees(pitch):.2f} degrees")
print(f"Roll: {np.degrees(roll):.2f} degrees")
