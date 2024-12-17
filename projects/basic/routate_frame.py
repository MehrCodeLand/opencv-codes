import cv2
import numpy as np

# Load the image
image = cv2.imread('bekham1.jpg')

# Get the dimensions of the image
(h, w) = image.shape[:2]

# Define the center of the image
center = (w // 2, h // 2)

# Define the rotation matrix
angle = 45  # Angle in degrees
scale = 1.0  # Scale factor
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation to the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
cv2.imwrite('bekahm3_rotaite.jpg' , rotated_image)

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
