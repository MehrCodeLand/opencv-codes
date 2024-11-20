import cv2 as cv
import pandas as pd
import numpy as np



image = cv.imread('images/one.png')
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

r , thresh = cv.threshold(gray , np.mean(gray) , 255 , cv.THRESH_BINARY_INV)
contoures, h = cv.findContours(thresh , cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)

cnt = sorted(contoures , key=cv.contourArea)[-1]


mask = np.zeros((image.shape[0] , image.shape[1]) , dtype='uint8')
masked = cv.drawContours(mask , [cnt] , 0 , (255,255,255) , -1)

final = cv.bitwise_and(image , image , mask=masked)
cv.imshow('image' , final)
cv.waitKey()
cv.destroyAllWindows()