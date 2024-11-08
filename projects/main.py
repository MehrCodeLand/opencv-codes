import cv2 as cv 
from PIL import Image
from utill import get_limits

yellow = [0 , 255 , 255]
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    hsvImage = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
    
    lower , upper = get_limits(color=yellow)

    mask = cv.inRange(hsvImage, lower , upper)
    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox()
    
    if bbox is not None :
        x1 , y1 , x2 , y2 = bbox 
        
        cv.rectangle(frame , (x1,y1) , (x2,y2) , (0 , 255,0), 5)
    
    cv.imshow('frame' , frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()