import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Helper function to extract face bounding box
def get_face_bbox(image, detection):
    h, w, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    x1 = int(bboxC.xmin * w)
    y1 = int(bboxC.ymin * h)
    x2 = int((bboxC.xmin + bboxC.width) * w)
    y2 = int((bboxC.ymin + bboxC.height) * h)
    return x1, y1, x2, y2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables
prev_gray = None
prev_points = None
trajectory = []  # Store all points for persistent tracing
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for selfie-view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Every 10 seconds, detect head and reinitialize tracking points
    if time.time() - start_time > 10 or prev_points is None:
        start_time = time.time()
        results = face_detection.process(rgb_frame)
        if results.detections:
            detection = results.detections[0]
            x1, y1, x2, y2 = get_face_bbox(frame, detection)

            # Initialize tracking points within the bounding box
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
            # Add initial points to the trajectory
            for pt in prev_points:
                trajectory.append([pt.ravel()])
    else:
        # Continue tracking using Optical Flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

        # Update trajectory with new points
        for i, (new, old) in enumerate(zip(next_points, prev_points)):
            if status[i]:
                trajectory[i].append(new.ravel())
        
        # Update previous points and frame
        prev_points = next_points
        prev_gray = curr_gray

    # Draw all traced trajectories
    for points in trajectory:
        for j in range(1, len(points)):
            cv2.line(frame, tuple(points[j - 1].astype(int)), tuple(points[j].astype(int)), (0, 255, 0), 2)
            cv2.circle(frame, tuple(points[j].astype(int)), 2, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Persistent Head and Object Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
