import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load a video file
cap = cv2.VideoCapture(0)

# Create a MediaPipe Face Detection object
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = face_detection.process(rgb_frame)

        # Draw detections on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Display the frame
        cv2.imshow("Video Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
cv2.destroyAllWindows()
