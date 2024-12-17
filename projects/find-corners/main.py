import cv2
import numpy as np

img = cv2.imread("davidmask.JPG" , cv2.IMREAD_GRAYSCALE)

maxCorners = 100
qualityLevel = 0.1
minDistance = 10

corners = cv2.goodFeaturesToTrack(img , maxCorners, qualityLevel, minDistance)

corners = np.int0(corners)

for corner in corners:
    x ,y = corner.ravel()
    cv2.circle(img , (x,y), 5, 255, -1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
