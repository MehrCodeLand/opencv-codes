import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define function to calculate head pose
def get_head_pose(image, landmarks):
    # 3D model points of facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])
    
    # 2D image points of facial landmarks
    image_points = np.array([
        (landmarks[1].x * image.shape[1], landmarks[1].y * image.shape[0]),
        (landmarks[152].x * image.shape[1], landmarks[152].y * image.shape[0]),
        (landmarks[226].x * image.shape[1], landmarks[226].y * image.shape[0]),
        (landmarks[446].x * image.shape[1], landmarks[446].y * image.shape[0]),
        (landmarks[57].x * image.shape[1], landmarks[57].y * image.shape[0]),
        (landmarks[287].x * image.shape[1], landmarks[287].y * image.shape[0])
    ], dtype="double")

    # Camera internals (assuming a focal length of 1)
    focal_length = image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # SolvePnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Convert rotation matrix to Euler angles
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    return yaw, pitch, roll

# Function to convert rotation matrix to Euler angles
def rotation_matrix_to_euler_angles(R):
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

# Load an example image
image = cv2.imread("path_to_your_image.jpg")

# MediaPipe Face Mesh for detecting facial landmarks
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            yaw, pitch, roll = get_head_pose(image, face_landmarks.landmark)
            print(f"Yaw: {np.degrees(yaw):.2f}, Pitch: {np.degrees(pitch):.2f}, Roll: {np.degrees(roll):.2f}")

# Show the image (optional)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
