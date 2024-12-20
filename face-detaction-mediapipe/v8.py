import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def select_random_frame(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select a random frame number
    random_frame_number = random.randint(0, total_frames - 1)

    # Set the video to the random frame
    video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    # Read the frame
    ret, frame = video.read()

    # Release the video capture object
    video.release()

    # Check if the frame was successfully read
    if ret:
        return frame
    else:
        raise ValueError("Couldn't read the frame from the video.")

# Example usage
video_path = 'path_to_your_video_file.mp4'
random_frame = select_random_frame(video_path)


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
def detect_head_movement(yaw_angle, threshold= 2):
    if yaw_angle > threshold:
        return "Right"
    elif yaw_angle < -threshold:
        return "Left"
    return "Still"

# Function to Detect Head Pose
def detect_head_pose(landmarks, image_width, image_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),    # Nose tip
        (0.0, -330.0, -65.0), # Chin
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0), # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),  # Nose tip
        (landmarks[152][0], landmarks[152][1]), # Chin
        (landmarks[33][0], landmarks[33][1]),  # Left eye corner
        (landmarks[263][0], landmarks[263][1]),  # Right eye corner
        (landmarks[61][0], landmarks[61][1]),  # Left mouth corner
        (landmarks[291][0], landmarks[291][1])  # Right mouth corner
    ], dtype="double")

    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw_angle = np.degrees(rotation_matrix[1][0])
        return yaw_angle
    return None


def is_mouth_moving(mouth_landmarks, previous_mouth_distance, threshold=2.0):
    # Check if there are enough landmarks
    if len(mouth_landmarks) < 4:
        return False, previous_mouth_distance

    # Calculate upper and lower lip distances
    upper_lip = np.mean([mouth_landmarks[i] for i in range(2)], axis=0)
    lower_lip = np.mean([mouth_landmarks[i] for i in range(2, 4)], axis=0)

    # Compute mouth distance and detect movement
    current_mouth_distance = dist.euclidean(upper_lip, lower_lip)
    if abs(current_mouth_distance - previous_mouth_distance) > threshold:
        return True, current_mouth_distance
    return False, current_mouth_distance


# Initialize variables
previous_mouth_distance = 0.0
blink_threshold = 0.2
blink_detected = False

# Main Function
cap = cv2.VideoCapture(0)

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

            # Mouth Movement Detection
            mouth_landmarks = [landmarks[i] for i in [13, 14, 17, 18]]
            mouth_moving, previous_mouth_distance = is_mouth_moving(mouth_landmarks, previous_mouth_distance)

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
            cv2.putText(frame, f"Speaking: {'Yes' if mouth_moving else 'No'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
