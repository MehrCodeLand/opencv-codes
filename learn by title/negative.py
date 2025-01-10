import cv2
import numpy as np

# Read the image
image = cv2.imread('./images/one.png', cv2.IMREAD_GRAYSCALE)

# Get the maximum intensity value (L-1)
L = 256  # For 8-bit images
negative_image = L - 1 - image  # Apply the formula


cv2.imwriteq('negative_image.png', negative_image)
# Save or display the negative image
cv2.imshow('Original Image', image)
cv2.imshow('Negative Image', negative_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
