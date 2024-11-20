import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Initialize OpenCV and Dlib
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Real-time Face Detection and Anti-Spoofing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        # Extract eye landmarks
        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]

        # Calculate EAR for each eye
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Anti-spoofing: EAR below threshold indicates blinking
        if ear < 0.25:  
            cv2.putText(frame, "Real Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Fake Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw face rectangle
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
