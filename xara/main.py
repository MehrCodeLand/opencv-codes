import cv2
import numpy as np
import os

def separate_objects(image_path, output_folder):
    # Step 1: Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Preprocess the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.RETR_TREE)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} objects.")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True) 
    
    # Step 4: Extract and save each object
    for idx, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the object from the original image
        object_img = image[y:y+h, x:x+w]
        
        # Save the extracted object
        output_path = os.path.join(output_folder, f"object_{idx+1}.png")
        cv2.imwrite(output_path, object_img)
        print(f"Saved object {idx+1} to {output_path}")
    
    # Display the result with bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
image_path = 'xara/img/one.png'
output_folder = 'xara/img'


separate_objects(image_path, output_folder)
