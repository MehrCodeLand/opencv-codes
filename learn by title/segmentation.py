import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('images/3.jpeg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply thresholding
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Remove small noise with morphological operations
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
_, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply the Watershed algorithm
markers = cv.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

# Display results
cv.imshow('Original Image', image)
cv.imshow('Binary Image', binary)
cv.imshow('Watershed Segmentation', image)

cv.waitKey(0)
cv.destroyAllWindows()
