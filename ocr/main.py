import cv2 as cv
from PIL import Image
import numpy as np
import pytesseract


im_file = "hello.jpeg"
im = Image.open(im_file)

# invert image
image = cv.imread('hello.jpeg')
invert_image = cv.bitwise_not(image)
image_gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
thresh, im_bw = cv.threshold(image_gray, 170 , 255 , cv.THRESH_BINARY)
 
# noise remove 
kernel = np.ones((1,1), np.uint8)
nois_image = cv.dilate(im_bw, kernel , iterations=5)

cv.imshow('bitwise' , nois_image)
cv.waitKey(0)
cv.destroyAllWindows()



# _, binary_image = cv.threshold(image_gray , 128 , 255 , cv.THRESH_BINARY)
# binary_image = cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# _, binary_image = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow('hello ' , binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()


