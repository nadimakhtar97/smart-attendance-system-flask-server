import cv2
from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import base64
import io
import re
from recognition import recognize
from training import train
from face_dataset import collect_dataset

app = Flask(__name__)
CORS(app)


# Take in base64 string and return PIL image
def store_dataset(images, face_id):
    count = 0
    for base64_string in images:
        count += 1
        base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
        image = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image))
        image.save("dataset/User." + str(face_id) + '.' + str(count) + ".jpg")
        image = cv2.imread("dataset/User." + str(face_id) + '.' + str(count) + ".jpg")
        collect_dataset(image, face_id, count)


def string_to_image(base64_string):
    base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    image = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image))
    image.save('images/test.jpg')
    return image


@app.route('/train', methods=['GET'])
def train_dataset():
    train()
    return {"status": "training successful"}


@app.route('/img', methods=['POST'])
def process_image():
    data = request.get_json()
    images = data["base64Images"]
    face_id = data["rollNo"]
    store_dataset(images, face_id)
    train()
    return {
        "status": "image processing successful",
    }


@app.route('/identify', methods=['POST'])
def identify():
    data = request.get_json()
    base64_string = data["base64image"]
    string_to_image(base64_string)
    image = cv2.imread('images/test.jpg')
    result = recognize(image)
    return {
        "status": "students recognition successful",
        "id": result
    }


if __name__ == '__main__':
    app.run(debug=True)
