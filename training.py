import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# function to get the images and label data
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for imagePath in image_paths:

        pil_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')

        # print("this is id ", os.path.split(imagePath)[-1].split(".")[1])
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = classifier.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids


def train():
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    # print(ids)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))


# train()
