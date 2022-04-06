import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def recognize(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    print(faces)

    if len(faces) == 0:
        return -2

    for (x, y, w, h) in faces:

        _id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print("this is confidence ", _id, confidence)
        if confidence < 70:
            confidence = "  {0}%".format(round(100 - confidence))
            print(confidence)
            return _id
        else:
            _id = -1
            confidence = "  {0}%".format(round(100 - confidence))
            print(confidence)
            return _id


# z = cv2.imread('images/shezan1.jpeg')
# cv2.imshow('image', z)
# cv2.waitKey(0)
# recognize(z)
# print("\n [INFO] Exiting Program")
# cv2.destroyAllWindows()
