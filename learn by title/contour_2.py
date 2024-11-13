import cv2 as cv 



image = cv.imread('images/one.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, thresh = cv.threshold(gray, 200, 255, 0)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv.moments(cnt)


cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

area = cv.contourArea(cnt)

perimeter = cv.arcLength(cnt,True)

epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)

# Draw contours on the original image
cv.drawContours(image, contours, -1, (0, 255, 0), 2)

cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()