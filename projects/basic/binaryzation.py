import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (800, 600))

# Apply global thresholding
_, binary_global = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding
binary_adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
binary_adaptive_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)

# Display the original and binary images
cv2.imshow('Original Image', image)
cv2.imshow('Global Thresholding', binary_global)
cv2.imshow('Adaptive Thresholding (Mean)', binary_adaptive_mean)
cv2.imshow('Adaptive Thresholding (Gaussian)', binary_adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
