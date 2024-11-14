from pdf_2_image import pdf_to_jpg
import numpy as np
import os
import easyocr
import cv2 as cv


images_names = []
images_ocr = []


def read_image_name():
    for file_name in os.listdir('images'):
        if os.path.isfile(os.path.join('images' , file_name)):
            images_names.append(file_name)


    return images_names 


def show_image_bbox():

    for texts in images_ocr:
        for t in texts:
            bbox , text , score = t            
            if score > 0.35:
                print(text)



def main_ocr():
    images_name = read_image_name()

    reader = easyocr.Reader(['en'] , gpu=False)

    for image_name in images_name:
        print(f'../images/{image_name}')
        image = cv.imread(f'images/{image_name}')

        text = reader.readtext(image)
        images_ocr.append(text)


    show_image_bbox()


main_ocr()