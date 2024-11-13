import cv2 as cv
import numpy as np
 
image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)


kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(gray , kernel , iterations=1)

dilation = cv.dilate(gray , kernel , iterations=1)

opening = cv.morphologyEx(gray , cv.MORPH_OPEN , kernel )

closing = cv.morphologyEx( image, cv.MORPH_CLOSE , kernel)

gradiant = cv.morphologyEx(gray , cv.MORPH_GRADIENT , kernel)

tophat = cv.morphologyEx(gray , cv.MORPH_TOPHAT , kernel)

blackhat = cv.morphologyEx(gray , cv.MORPH_BLACKHAT , kernel)


cv.imshow('erode' , erosion)
cv.imshow('dilation' , dilation)
cv.imshow('opening' , opening)
cv.imshow('closing' , closing)
cv.imshow('gradiant' , gradiant)
cv.imshow('tophat' , tophat)
cv.imshow('blackhat' , blackhat)
cv.waitKey()
cv.destroyAllWindows()