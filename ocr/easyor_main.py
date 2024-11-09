import easyocr as eo
import cv2 as cv


reader = eo.Reader(['en'])

result = reader.readtext('images/one.png')

print(result)