import cv2 as cv
from PIL import Image
import pytesseract


im_file = "hello.jpeg"
im = Image.open(im_file)

# image = cv.imread('hello.jpeg')
# image_gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
# _, binary_image = cv.threshold(image_gray , 128 , 255 , cv.THRESH_BINARY)
# binary_image = cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# _, binary_image = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow('hello ' , binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()


