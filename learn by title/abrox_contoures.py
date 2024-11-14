import cv2
import numpy as np

# Load the image
image = cv2.imread('images/one.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Contour approximation
epsilon = 0.02 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)

# Draw original contour
cv2.drawContours(gray, [contours[0]], -1, (0, 255, 0), 2)

# Draw approximated contour
cv2.drawContours(gray, [approx], -1, (255, 0, 0), 2)

# Display the result
cv2.imshow('Contour Approximation', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
