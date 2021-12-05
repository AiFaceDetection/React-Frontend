from flask import Flask, request
from flask_cors import CORS
import json
from werkzeug import datastructures
from face_rec import FaceRec, taeshin
from PIL import Image
import base64
import io
import os
import shutil
import time

# import for face identification
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import cv2
import numpy as np
from itertools import chain


app = Flask(__name__)
CORS(app)

# face identification var
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")


@app.route('/api', methods=['POST', 'GET'])
def api():
    data = request.get_json()
    resp = 'undetected'
    directory = './face'

    if data:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        
        if not os.path.exists(directory):
            try:
                os.mkdir(directory)
                time.sleep(1)
                result = data['data']
                b = bytes(result, 'utf-8')
                image = b[b.find(b'/9'):]
                im = Image.open(io.BytesIO(base64.b64decode(image)))
                im.save(directory + '/face.png')

                full_file_path = os.path.join(face_dir , 'face.jpg')
                predictions = predict(full_file_path, model_path="trained_knn_model.clf")
                resp = []
                try:
                    for name, (top, right, bottom, left) in predictions:
                        resp.append(name)
                finally:
                    if len(resp) == 0:
                        resp= ["Face not detected"]
            except:
                pass

    return resp[0]


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.48):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    try:
        # Load image file and find face locations
        X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    except:
        pass

if __name__ == '__main__':
	 app.run(debug=True)