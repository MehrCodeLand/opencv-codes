import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)

# Parameters for goodFeaturesToTrack
maxCorners = 100
qualityLevel = 0.01
minDistance = 10

# Detect corners using goodFeaturesToTrack
corners = cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)

# Convert corners to integer
corners = np.int0(corners)

# Draw circles around detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Display the result
cv2.imshow('Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
