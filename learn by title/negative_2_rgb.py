import cv2
import numpy as np

# Load the color negative image
color_negative = cv2.imread('./images/OIP.jpeg')

# Check if the image was loaded successfully
if color_negative is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Define the maximum intensity for an 8-bit image
    L = 256  # Maximum intensity (0-255 for each channel)

    # Convert the color negative back to the original color image
    original_image = L - 1 - color_negative

    # Display the color negative and the restored original image
    cv2.imshow('Color Negative', color_negative)
    cv2.imshow('Restored Original Image', original_image)

    # Save the restored image if needed
    cv2.imwrite('restored_original_image.jpg', original_image)

    # Wait for user input and close the display windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
