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
        if not ret:
            break

        # Call the emotion detection method
        emotion_detection_webcam(frame)

        cv.imshow('captures', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

def load_images():
    user_images = []
    for filename in os.listdir('known'):
        if filename.endswith(('.jpeg', '.jpg')):
            image_path = os.path.join('known', filename)
            image = cv.imread(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            user_images.append(encoding)
    return user_images

def compare_image(frame, local_user_images):
    captured_encoding = face_recognition.face_encodings(frame)[0]
    for user_image in local_user_images:
        match = face_recognition.compare_faces([user_image], captured_encoding)
        if match[0]:
            return True
    return False

def emotion_face_detection(frame):
    detector = FER()
    emotions = detector.detect_emotions(frame)

    if emotions:
        emotion, score = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
        print(f"Detected emotion: {emotion} with score {score}")

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
        print(f"Head direction: {direction}")

def emotion_detection_webcam(frame):
    detector = FER()

    # Detect emotions in the frame
    emotions = detector.detect_emotions(frame)

    # {'box': array([166, 124, 293, 293]),
    #  'emotions': {'angry': 0.01, 'disgust': 0.0, 'fear': 0.01, 'happy': 0.0, 'sad': 0.05, 'surprise': 0.0,
    #               'neutral': 0.92}}

    if emotions:
        for face in emotions:
            (x, y, w, h) = face['box']
            emotion, score = max(face['emotions'].items(), key=lambda item: item[1])

            # Draw a rectangle around the face
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Put emotion label and score
            cv.putText(frame, f"{emotion}: {score:.2f}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

open_webcam()
