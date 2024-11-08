import cv2 as cv

# Load image
image = cv.imread("one.png", cv.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred_image = cv.GaussianBlur(image, (5, 5), sigmaX=1.0)

# Display result
cv.imshow("Gaussian Blur", blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()
