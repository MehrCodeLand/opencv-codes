import cv2 as cv
import numpy as np

image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

# avaraging
kernel = np.ones((5,5) , np.float32)/25
dst = cv.filter2D(image , -1 , kernel)

# blure
blure = cv.blur(image , (5,5))

# gaussion blure
g_blure = cv.GaussianBlur(image , (5,5) , 0)

# median blurring
median = cv.medianBlur(image , 5)

# Bilateral Filtering
bf = cv.bilateralFilter(image , 9 ,75 , 75)


cv.imshow('Bilateral Filtering' , bf)
cv.imshow('median' , median)
cv.imshow('g-blure' , g_blure)
cv.imshow('blure' ,blure)
cv.imshow('averaging' , dst)
cv.waitKey()
cv.destroyAllWindows()
