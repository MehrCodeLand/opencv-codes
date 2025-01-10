

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('./images/one.png', cv2.IMREAD_GRAYSCALE)

# Identify the intensity range for the line
# (Manually or via histogram analysis, e.g., line intensity = [100, 150])
line_min, line_max = 100, 150

# Custom intensity scaling
scaled_image = np.zeros_like(image)
mask = (image >= line_min) & (image <= line_max)
scaled_image[mask] = ((image[mask] - line_min) / (line_max - line_min) * 255).astype(np.uint8)

# Apply thresholding to isolate the line
_, thresholded_image = cv2.threshold(scaled_image, 50, 255, cv2.THRESH_BINARY)

# Optional: Morphological operations to refine the line
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
refined_line = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Display results
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Scaled Image")
plt.imshow(scaled_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Refined Line")
plt.imshow(refined_line, cmap='gray')

plt.show()
