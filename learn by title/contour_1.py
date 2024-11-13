import cv2 as cv

image = cv.imread('images/one.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, thresh = cv.threshold(gray, 150, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(image, contours, -1, (0, 255, 0), 2)

cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()

# *** tips ***
# Contours are curves joining all the continuous points along the 
# boundary, having the same color or intensity. In simpler terms,
# contours are the outlines or edges of objects within an image

