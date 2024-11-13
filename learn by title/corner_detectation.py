import cv2 as cv 
import pandas as pd
import numpy as np 


image = cv.imread('images/one.png')
orginal_image = image
orginal_image2 = image

image = cv.cvtColor(image , cv.COLOR_BGR2GRAY)


# Shi-Tomasi Corner Detector & Good Features to Track
corners = cv.goodFeaturesToTrack(image , 100 , 0.01 , 10 , -1 )
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    print(corner)
    cv.circle(image , (x,y), 3 , (0,0,255) )

# Harris Corner Detector in OpenCV
gray = image
gray = np.float32(gray)

dst = cv.cornerHarris(gray , 2,3, 0.04)
dst = cv.dilate(dst,None)
ret , dst = cv.threshold(dst,0.01*dst.max(), 255 , 0)
dst = np.uint8(dst)

ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
orginal_image2[dst>0.01*dst.max()] = [0 , 0 , 255]

res = np.hstack((centroids,corners))
res = np.int0(res)

orginal_image2[res[:,1],res[:,0]]=[0,0,255]
orginal_image2[res[:,3],res[:,2]] = [0,255,0]

cv.imshow('image' , orginal_image)
cv.imshow('hello' , image)
cv.imshow('hhh' , orginal_image2)

cv.waitKey(0)
cv.destroyAllWindows()



# extra info 
# Difference Between SubPixel Accuracy and Harris Corner Detection
# Harris Corner Detection: Detects corners based on the windowed
# intensity gradient covariance matrix. It provides an initial estimate 
# of corner locations1.
# SubPixel Accuracy: Refines the initial corner locations detected
# by Harris Corner Detection to achieve higher precision. It iteratively
# adjusts the corner positions to subpixel accuracy1.


# article : https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html