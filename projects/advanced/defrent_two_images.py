import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def resize_images(image1, image2):
    # Resize second image to match the dimensions of the first image
    resized_image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return resized_image2


def compare_images(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same dimensions
    gray_image2 = resize_images(gray_image1, gray_image2)

    # Calculate SSIM between the two images
    score, diff = ssim(gray_image1, gray_image2, full=True)
    ssim_score = score

    # Calculate MSE between the two images
    mse = np.mean((gray_image1 - gray_image2) ** 2)

    # Convert SSIM score to a percentage difference
    percentage_difference = (1 - ssim_score) * 100

    return percentage_difference, mse


# Load the two images
image1 = cv2.imread('bekham1.jpg')
image2 = cv2.imread('bekham1.jpg')

# Compare the images
percentage_difference, mse = compare_images(image1, image2)

print(f"Percentage Difference: {percentage_difference:.2f}%")
print(f"Mean Squared Error: {mse:.2f}")

# Display the images for visual inspection
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
