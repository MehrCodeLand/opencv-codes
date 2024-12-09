import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

# Load an example image
image = cv2.imread("images/3.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect pose landmarks
results = pose.process(image_rgb)

# Draw the landmarks on the image
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

# Display the image with landmarks
cv2.imshow("Pose Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
