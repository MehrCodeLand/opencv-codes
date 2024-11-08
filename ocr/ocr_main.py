import cv2 as cv
import pandas as pd 
import numpy as np

image = cv.imread('hello.jpeg')
# gray_image = cv.cvtColor(image , cv.COLOR_BAYER_BG2GRAY)

resize_image = cv.resize(image , (500 , 500))

cv.imshow('image' , resize_image)
cv.waitKey(0)
cv.destroyAllWindows()