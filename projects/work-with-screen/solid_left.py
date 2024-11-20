import cv2 as cv

# Capture video from webcam
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Fill the left half of the frame with black
    frame[:, :width // 2] = (0, 0, 0)

    # Display the frame
    cv.imshow('Video Capture', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
