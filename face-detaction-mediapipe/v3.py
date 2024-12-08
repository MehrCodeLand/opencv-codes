import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, indices):
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

# Function to estimate head pose based on key landmarks
def estimate_head_pose(landmarks, image_width, image_height):
    nose_tip = [landmarks[1].x * image_width, landmarks[1].y * image_height]
    left_eye = [landmarks[33].x * image_width, landmarks[33].y * image_height]
    right_eye = [landmarks[263].x * image_width, landmarks[263].y * image_height]

    # Simple heuristic: if the difference in position between frames is too small, it's likely fake
    head_movement = abs(left_eye[0] - right_eye[0])
    return head_movement

# Variables for temporal analysis
frame_count = 0
blink_count = 0
closed_eye_frames = 0
open_eye_frames = 0

cap = cv2.VideoCapture(0)

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
                # EAR for blink detection
                left_eye_ear = calculate_ear(face_landmarks.landmark, [33, 133, 159, 145, 160, 144])
                right_eye_ear = calculate_ear(face_landmarks.landmark, [362, 263, 386, 374, 387, 373])
                average_ear = (left_eye_ear + right_eye_ear) / 2

                # Detect blinking based on EAR
                if average_ear < 0.25:  # Threshold for closed eyes
                    closed_eye_frames += 1
                    open_eye_frames = 0
                else:
                    open_eye_frames += 1
                    if closed_eye_frames > 3:  # Blink detected (adjust threshold as needed)
                        blink_count += 1
                        closed_eye_frames = 0

                # Head movement analysis
                head_movement = estimate_head_pose(face_landmarks.landmark, image_width, image_height)
                if head_movement < 1:  # Minimal movement indicates a flat image
                    cv2.putText(frame, "Fake Detected: No Head Movement", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif blink_count > 1:
                    cv2.putText(frame, "Live Face Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Fake Detected: No Blinking", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw facial landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Display the results
        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
cv2.destroyAllWindows()
