import cv2 as cv
import numpy as np
import face_recognition
import os

cam = cv.VideoCapture(0)


def open_webcam():
    while True:
        ret, frame = cam.read()

        count = 1
        if cv.waitKey(1) == ord('c'):
            local_user_images = load_images()

            res = compare_image(frame , local_user_images)
            print(res)
        cv.imshow('captures', frame)

        if cv.waitKey(1) == ord('q'):
            break


def login_user(user_image):
    user_image = cv.cvtColor(user_image, cv.COLOR_BGR2GRAY)


    return False


# this method convert image to encode format ( local images)
def load_images():
    user_images = []
    for filename in os.listdir('known'):
        if filename.endswith(('.jpeg' , '.jpg')):
            image_path = f'./known/{filename}'
            image = cv.imread(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            user_images.append(encoding)
    return user_images

def compare_image(frame , local_user_images):
    captured_encoding = face_recognition.face_encodings(frame)[0]
    for user_image in local_user_images:
        match = face_recognition.compare_faces([user_image] , captured_encoding)
        print(match)
        if match[0]:
            return True
        else:
            return False

open_webcam()

cv.destroyAllWindows()
