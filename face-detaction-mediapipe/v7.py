import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR Calculation Function
def calculate_ear(eye_landmarks):
    vertical_1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Eye-Lip Distance Function
def calculate_eye_lip_distance(eye_center, lip_center):
    return dist.euclidean(eye_center, lip_center)

# Head Movement Detection
def detect_head_movement(yaw_angle, threshold=10):
    if yaw_angle > threshold:
        return "Right"
    elif yaw_angle < -threshold:
        return "Left"
    return "Still"

# Function to Detect Head Pose
def detect_head_pose(landmarks, image_width, image_height):
    # 3D model points for face landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),    # Nose tip
        (0.0, -330.0, -65.0), # Chin
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0), # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    
    # 2D image points
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),  # Nose tip
        (landmarks[152][0], landmarks[152][1]), # Chin
        (landmarks[33][0], landmarks[33][1]),  # Left eye corner
        (landmarks[263][0], landmarks[263][1]),  # Right eye corner
        (landmarks[61][0], landmarks[61][1]),  # Left mouth corner
        (landmarks[291][0], landmarks[291][1])  # Right mouth corner
    ], dtype="double")

    # Camera matrix and distortion coefficients
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # SolvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw_angle = np.degrees(rotation_matrix[1][0])
        return yaw_angle
    return None

# Main Function
cap = cv2.VideoCapture(0)
prev_face_bbox = None
frame_count = 0
blink_threshold = 0.2
blink_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = [(int(pt.x * frame_width), int(pt.y * frame_height)) for pt in face_landmarks.landmark]

            # EAR Calculation (Left Eye: 33-133, Right Eye: 362-263)
            left_eye = [landmarks[i] for i in [33, 159, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 386, 263, 373, 380]]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Eye-Lip Distance
            eye_center = np.mean([landmarks[33], landmarks[263]], axis=0)
            lip_center = np.mean([landmarks[61], landmarks[291]], axis=0)
            eye_lip_distance = calculate_eye_lip_distance(eye_center, lip_center)

            # Head Pose Estimation
            yaw_angle = detect_head_pose(landmarks, frame_width, frame_height)

            # Check Blink
            if ear < blink_threshold:
                blink_detected = True
            else:
                blink_detected = False

            # Check Head Movement
            movement = detect_head_movement(yaw_angle)

            # Draw Results
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye-Lip Dist: {eye_lip_distance:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Movement: {movement}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
