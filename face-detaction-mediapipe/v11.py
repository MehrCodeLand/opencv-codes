import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define helper functions
def detect_mask(landmarks):
    """
    Detect if the user is wearing a mask by checking key facial landmarks around the nose and mouth.
    """
    # Landmarks around the nose and mouth
    mouth_landmarks = [13, 14, 78, 308, 312, 402, 13, 82, 178, 87, 317]  # Outer boundary of the mouth
    nose_landmarks = [1, 2, 4, 6, 94, 122, 49, 168]  # Landmarks on the nose

    # Calculate bounding boxes for mouth and nose areas
    mouth_x = [landmarks[pt].x for pt in mouth_landmarks]
    mouth_y = [landmarks[pt].y for pt in mouth_landmarks]
    nose_x = [landmarks[pt].x for pt in nose_landmarks]
    nose_y = [landmarks[pt].y for pt in nose_landmarks]

    # Thresholds for detecting masks
    visibility_threshold = 0.5  # Visibility cutoff
    occlusion_threshold = 0.7  # Adjust based on empirical testing

    # Check visibility of key landmarks
    visible_mouth_points = [landmarks[pt].visibility > visibility_threshold for pt in mouth_landmarks]
    visible_nose_points = [landmarks[pt].visibility > visibility_threshold for pt in nose_landmarks]

    # If mouth and nose areas are nearly occluded, assume a mask
    mouth_occluded = sum(visible_mouth_points) / len(visible_mouth_points) < occlusion_threshold
    nose_occluded = sum(visible_nose_points) / len(visible_nose_points) < occlusion_threshold

    # Final decision: mask is present if both mouth and nose are occluded
    return mouth_occluded and nose_occluded

def detect_corona_medicine_mask(landmarks):
    """
    Detect the type of mask the user is wearing (corona mask, medicine mask, or no mask).
    """
    # Check if a mask is present
    if not detect_mask(landmarks):
        return "No Mask"

    # Analyze patterns for corona and medicine masks
    nose_landmarks = [1, 2, 4, 6, 94, 122, 49, 168]
    nose_x = [landmarks[pt].x for pt in nose_landmarks]
    nose_y = [landmarks[pt].y for pt in nose_landmarks]

    nose_width = max(nose_x) - min(nose_x)
    nose_height = max(nose_y) - min(nose_y)

    # Heuristic to distinguish mask types
    if nose_width > 0.05 and nose_height < 0.03:  # Example values, adjust as needed
        return "Corona Mask"
    elif nose_width < 0.05 and nose_height > 0.03:
        return "Medicine Mask"

    return "Unknown Mask"

def detect_lower_face_covered(landmarks):
    """
    Detect if the lower part of the face (from the nose to the chin) is covered or obscured.
    """
    # Landmarks from the nose down to the chin
    lower_face_landmarks = [1, 2, 4, 6, 94, 122, 49, 168, 17, 18, 200, 152, 175, 396]

    # Check visibility of key landmarks
    visibility_threshold = 0.5
    visible_points = [landmarks[pt].visibility > visibility_threshold for pt in lower_face_landmarks]

    # If most points are not visible, lower face is covered
    if sum(visible_points) / len(visible_points) < 0.5:
        return True

    return False

def detect_obscured_face(landmarks):
    """
    Detect if the face from the nose to the lower chin is entirely obscured by any object.
    """
    # Define the key landmarks for the lower face region
    lower_face_landmarks = [1, 2, 4, 6, 94, 122, 49, 168, 17, 18, 200, 152, 175, 396]
    
    # Calculate bounding boxes for the lower face region
    face_x = [landmarks[pt].x for pt in lower_face_landmarks]
    face_y = [landmarks[pt].y for pt in lower_face_landmarks]

    # Check if the area is fully occluded
    visibility_threshold = 0.5
    visible_points = [landmarks[pt].visibility > visibility_threshold for pt in lower_face_landmarks]

    if sum(visible_points) / len(visible_points) < 0.5:
        return True  # The face is not clear

    return False

# Start the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )

            # Perform mask detection
            is_masked = detect_mask(face_landmarks.landmark)
            mask_status = "Mask Detected" if is_masked else "No Mask"

            # Determine the type of mask
            mask_type = detect_corona_medicine_mask(face_landmarks.landmark)

            # Check if the lower face is covered
            lower_face_covered = detect_lower_face_covered(face_landmarks.landmark)
            if lower_face_covered:
                warning_message = "Face Not Fully Visible"
                cv2.putText(frame, warning_message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if the lower face is obscured
            obscured_face = detect_obscured_face(face_landmarks.landmark)
            if obscured_face:
                cv2.putText(frame, "Lower Face Obscured", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display the results
            cv2.putText(frame, mask_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mask Type: {mask_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Face Detection and Mask Analysis', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
