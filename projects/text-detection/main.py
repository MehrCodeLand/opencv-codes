import easyocr
import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread('images.jpeg')
image = cv.cvtColor(image , cv.COLOR_BGR2RGB)

reader = easyocr.Reader(['en'] , gpu=False)
texts = reader.readtext(image)

for t in texts:
    bbox , text , score = t

    if score > 0.35:
        cv.rectangle(image ,bbox[0] , bbox[2] , (0 , 255 , 0) , 5)
        cv.putText(image , text , bbox[0] , cv.FONT_HERSHEY_COMPLEX , 0.65 , (255 , 0 , 0) , 2 )


plt.imshow(image)
plt.show()

# cv.imshow('image' , image)
# cv.waitKey()
# cv.destroyAllWindows()