import cv2 as cv
import numpy as np


image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray , 100 ,200 )

cv.imshow('canny' , edges) 
cv.waitKey()
cv.destroyAllWindows()