import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect Faces
    results = face_detection.process(rgb_frame)

    # Draw and Analyze Results
    if results.detections:
        for detection in results.detections:
            # Get Bounding Box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract ROI (Lower Half of Face)
            lower_face = frame[y + h//2:y + h, x:x + w]

            # Analyze Mask Features (Blur or Smoothness)
            # Convert to grayscale and calculate Laplacian variance
            gray_face = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()

            # Threshold for Mask Detection
            mask_detected = variance < 50  # Adjust based on testing

            # Display Results
            label = "Mask Detected" if mask_detected else "No Mask"
            color = (0, 255, 0) if mask_detected else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display Video
    cv2.imshow('Mask Detection', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
