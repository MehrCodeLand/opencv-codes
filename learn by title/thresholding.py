import cv2 as cv
import numpy as np 

image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)


# threshould - binary 
ret , th1 = cv.threshold(gray , 127 , 255 ,cv.THRESH_BINARY)

#threshold - adaptive mean
th2 = cv.adaptiveThreshold(gray, 255 , cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY,11,2)

# threshold - adaptive gaussian
th3 = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                           cv.THRESH_BINARY,11,2)

cv.imshow('threshould3' , th3 )
cv.imshow('threshold2'  , th2)
cv.imshow('threshold1'  , th1)
cv.waitKey()
cv.destroyAllWindows()


# 1. Thresholding
# Thresholding is a simple, yet effective, method of image 
# segmentation. It converts a grayscale image into a binary image,
# separating pixels into two groups based on a defined threshold value.

# 2. Adaptive Gaussian Thresholding
# Adaptive Gaussian Thresholding overcomes the
# limitations of simple thresholding by calculating
# the threshold value for smaller regions of the image.
# This method is useful for images with varying lighting conditions.

# 3. Adaptive Mean Thresholding
# Adaptive Mean Thresholding is similar to Adaptive Gaussian
# Thresholding but calculates the threshold value based on
# the mean of the pixel values in the neighborhood region.