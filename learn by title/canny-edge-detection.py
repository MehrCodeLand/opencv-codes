import cv2 as cv
import matplotlib.pyplot as plt 

image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

# Canny Edge Detection in OpenCV
edge = cv.Canny(gray , 100 , 200 ) 


cv.imshow('image' , edge)
cv.waitKey()
cv.destroyAllWindows()



# *** tips ***

# Use Cases of Canny Edge Detection
# Object Detection: Identifying the boundaries of objects within an image.
# Image Segmentation: Separating different parts of an image based on edges.
# Feature Extraction: Extracting important features from images for further analysis.
# Medical Imaging: Detecting edges in medical scans to identify anatomical structures.
# Autonomous Vehicles: Helping in lane detection and obstacle recognition.