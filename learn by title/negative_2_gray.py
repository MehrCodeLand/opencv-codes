import cv2

# Read the color negative image
negative_image = cv2.imread('./images/negative_image.png')

# Revert to the original color image
L = 256  # Maximum intensity for 8-bit images
original_image = L - 1 - negative_image

rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
cv2.imshow('Reverted Color Image', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
