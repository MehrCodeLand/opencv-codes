import random
from fer import FER
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

            emotion_face_detecion(frame)
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


def emotion_face_detecion(frame):

    detector = FER()
    emotions = detector.detect_emotions(frame)

    if emotions:
        emotions , score = max(emotions[0]['emotions'].items() , key=lambda  item: item[1])
        print(f"Detected emotion: {emotions} with score {score}")

    face_landmarks_list = face_recognition.face_landmarks(frame)
    for face_landmarks in face_landmarks_list:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']

        dx = np.mean([p[0] for p in right_eye]) - np.mean([p[0] for p in left_eye])
        if dx > 10:
            direction = "right"
        elif dx < -10:
            direction = "left"
        else:
            direction = "straight"
        print(f"head direction {direction}")


open_webcam()

cv.destroyAllWindows()
