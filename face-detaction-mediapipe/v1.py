import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, indices):
    # EAR formula: (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])

    # Compute EAR
    vertical_dist = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal_dist = 2 * np.linalg.norm(p1 - p4)
    ear = vertical_dist / horizontal_dist
    return ear

# Function to estimate head pose based on landmarks
def estimate_head_pose(landmarks, image_width, image_height):
    # Convert normalized coordinates to pixel coordinates
    def to_pixel(point):
        return int(point.x * image_width), int(point.y * image_height)

    # Landmark indices for key face points
    nose_tip = to_pixel(landmarks[1])  # Nose tip
    chin = to_pixel(landmarks[152])   # Chin
    left_eye = to_pixel(landmarks[33])  # Left eye corner
    right_eye = to_pixel(landmarks[263])  # Right eye corner
    left_mouth = to_pixel(landmarks[61])  # Left mouth corner
    right_mouth = to_pixel(landmarks[291])  # Right mouth corner

    # Basic estimation: differences in pixel locations
    dx = abs(left_eye[0] - right_eye[0])
    dy = abs(nose_tip[1] - chin[1])

    return dx, dy  # Example metric for stability (improve with external libraries)

# Load the video file
cap = cv2.VideoCapture(0)

# MediaPipe Face Mesh
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

        # Process detected landmarks for liveness detection
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # EAR for blink detection
                ear = calculate_ear(face_landmarks.landmark, [33, 133, 159, 145, 160, 144])
                if ear < 0.25:  # Adjust threshold for blink detection
                    cv2.putText(frame, "Liveness Detected: Blink", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Head pose estimation (basic example)
                dx, dy = estimate_head_pose(face_landmarks.landmark, image_width, image_height)
                if dx < 50 or dy < 50:  # Threshold for limited movement (indicates a flat image)
                    cv2.putText(frame, "Fake Detected: Limited Movement", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Texture analysis: Example for brightness
                brightness = np.mean(frame)
                if brightness < 50:  # Very dark images might be spoofed
                    cv2.putText(frame, "Warning: Low Texture Variance", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Video Face & Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
cv2.destroyAllWindows()
