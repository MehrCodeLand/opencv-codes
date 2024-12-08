import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define 3D model points of key facial landmarks (nose tip, eyes, mouth corners, etc.)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),       # Nose tip
    (-30.0, -125.0, -30.0), # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

# Camera matrix (assuming no lens distortion)
FOCAL_LENGTH = 1
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 0.5],
    [0, FOCAL_LENGTH, 0.5],
    [0, 0, 1]
], dtype="double")

DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, indices, image_width, image_height):
    eye = [(landmarks[idx].x * image_width, landmarks[idx].y * image_height) for idx in indices]
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate head pose
def estimate_head_pose(landmarks, image_width, image_height):
    # Map 2D facial landmarks to 3D model points
    image_points = np.array([
        (landmarks[1].x * image_width, landmarks[1].y * image_height),   # Nose tip
        (landmarks[152].x * image_width, landmarks[152].y * image_height), # Chin
        (landmarks[33].x * image_width, landmarks[33].y * image_height),   # Left eye
        (landmarks[263].x * image_width, landmarks[263].y * image_height), # Right eye
        (landmarks[78].x * image_width, landmarks[78].y * image_height),   # Left mouth corner
        (landmarks[308].x * image_width, landmarks[308].y * image_height)  # Right mouth corner
    ], dtype="double")

    # SolvePnP to estimate rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        CAMERA_MATRIX,
        DIST_COEFFS
    )
    return translation_vector, rotation_vector

# Load video or webcam
video_path = 0  # Replace with file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Variables for temporal analysis
blink_count = 0
previous_translation = None
head_movement_count = 0

# Eye indices for EAR calculation
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
BLINK_THRESHOLD = 0.25
BLINK_FRAMES = 3

# MediaPipe Face Mesh for detecting facial landmarks
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_height, image_width, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Blink detection
                left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES, image_width, image_height)
                right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES, image_width, image_height)
                ear = (left_ear + right_ear) / 2.0

                if ear < BLINK_THRESHOLD:
                    blink_count += 1
                else:
                    if blink_count >= BLINK_FRAMES:
                        cv2.putText(frame, "Blink Detected: Live", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    blink_count = 0

                # Head pose estimation
                translation_vector, rotation_vector = estimate_head_pose(face_landmarks.landmark, image_width, image_height)

                if previous_translation is not None:
                    movement = np.linalg.norm(translation_vector - previous_translation)
                    if movement > 0.1:  # Adjust threshold for real movement
                        head_movement_count += 1
                        cv2.putText(frame, "Head Movement Detected: Live", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                previous_translation = translation_vector

                # Nod detection (up and down movement on Y-axis)
                if abs(translation_vector[1]) > 20:  # Threshold for nodding
                    cv2.putText(frame, "Nod Detected: Live", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
cv2.destroyAllWindows()
