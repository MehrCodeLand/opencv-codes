import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 


image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

laplacian = cv.Laplacian(gray , cv.CV_64F)
soblex = cv.Sobel(gray, cv.CV_64F,1,0,ksize=5)
sobely= cv.Sobel(gray , cv.CV_64F,0,1,ksize=5)


cv.imshow('x' , soblex)
cv.imshow('y' , sobely)
cv.imshow('laplacian' , laplacian)

cv.waitKey()
cv.destroyAllWindows()