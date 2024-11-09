import easyocr
import cv2 as cv



def ocr_farsi():
    image = cv.imread('m.jpeg')
    
    reader = easyocr.Reader(['fa'])
    results = reader.readtext('ocr/m.jpeg')
    
    show_ocr_result(results)
    
def show_ocr_result(results):
    for (bbox , text , prob) in results:
        print("Info {:4f}: {} ".format(prob,text).upper())


ocr_farsi()
