import cv2 

img = cv2.imread('./images/nois.jpeg')

blurred_image = cv2.GaussianBlur(img, (3,3) , 0)


cv2.imshow('blurr' , blurred_image)
cv2.imshow('orginal' , img)
cv2.waitKey()
cv2.destroyAllWindows()