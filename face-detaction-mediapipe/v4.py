import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Drawing Utilities
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

    # Use the rotation vector to compute head movement
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_movement = np.linalg.norm(rotation_matrix - np.eye(3))
    return head_movement

# Load video or webcam
video_path = 0  # Replace with file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Variables for temporal analysis
frame_count = 0
blink_count = 0
closed_eye_frames = 0
open_eye_frames = 0

# MediaPipe Face Mesh for detecting facial landmarks
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image_height, image_width, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Head movement detection
                head_movement = estimate_head_pose(face_landmarks.landmark, image_width, image_height)

                if head_movement < 0.05:  # Adjust threshold for minimal movement
                    cv2.putText(frame, "Fake Detected: No Head Movement", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Live Face Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw facial landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Display the frame
        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
cv2.destroyAllWindows()
