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
from PIL import Image
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import numpy as np
from itertools import chain
from imageio import imread


app = Flask(__name__)
CORS(app)

# face identification var
HEIGHT = 1080
WIDTH = 1920
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")


@app.route('/api', methods=['POST', 'GET'])
def api():
    data = request.get_json()
    result = data['data']
    b = bytes(result, 'utf-8')
    image = b[b.find(b'/9'):]
    img = imread(io.BytesIO(base64.b64decode(image)))

    match = []


    predictions = predict("hi", img, model_path="trained_knn_model.clf")

    for name, (top, right, bottom, left) in predictions:
        match.append(name)

    if (len(match) == 0):
        match.append("unknown")

    resp = match[0]
    return resp



@app.route('/apii', methods=['POST', 'GET'])
def apii():
    data = request.get_json()
    result = data['data']
    b = bytes(result, 'utf-8')
    image = b[b.find(b'/9'):]
    img = imread(io.BytesIO(base64.b64decode(image)))

    img1 = img[0: 1080, 0: 700]
    img2 = img[0: 1080, 700: 1920]

    card_encoding = face_recognition.face_encodings(img1)
    face_encoding = face_recognition.face_encodings(img2)

    resp = str(face_recognition.compare_faces([card_encoding], face_encoding))

    return resp

    # result = data['data']
    # b = bytes(result, 'utf-8')
    # image = b[b.find(b'/9'):]
    # imge = imread(io.BytesIO(base64.b64decode(image)))
    
    # predictions = predict("hi", imge, model_path="trained_knn_model.clf")

    # match1 = []
    # for name, (top, right, bottom, left) in predictions:
    #     match1.append(name)
    
    # if (len(match1) == 2 & match1[0] == match1[1]):
    #     resp = "True"
    # resp = "Flase"
    # return resp


    # try:
    #     card_image = face_recognition.load_image_file('card/card.jpg')
    #     face_image = face_recognition.load_image_file('face/face.jpg')

    #     card_encoding = face_recognition.face_encodings(card_image)[0]
    #     face_encoding = face_recognition.face_encodings(face_image)[0]

    #     result = str(face_recognition.compare_faces([card_encoding], face_encoding))
    # except:
    #     result  = "Face not detecteddd"

    # return result











    # resp = 'Face not detected'
    # directory = './getface'

    # if data:
    #     if os.path.exists(directory):
    #         shutil.rmtree(directory)
        
    #     # if not os.path.exists(directory):
    #     try:
    #         # os.mkdir(directory)
    #         time.sleep(1)
    #         result = data['data']
    #         b = bytes(result, 'utf-8')
    #         image = b[b.find(b'/9'):]
    #         img = imread(io.BytesIO(base64.b64decode(image)))
    #         im = Image.open(io.BytesIO(base64.b64decode(image)))
    #         im.save(directory + '/face.jpg')

    #         # frame = cv2.imread(directory + '/face.jpg')
    #         # card_frame = frame[HEIGHT-int(HEIGHT//1.4):int(HEIGHT//1.4), 0+20:int(40 * WIDTH // 100)-20]
    #         # face_frame = frame[0+20:HEIGHT-20, int(40 * WIDTH // 100)+20: int(40 * WIDTH // 100) + WIDTH - int(40 * WIDTH // 100)-20]
            
    #         # # card
    #         # cv2.imwrite(os.path.join(card_dir , 'card.jpg'), card_frame)
    #         # # face
    #         # cv2.imwrite(os.path.join(face_dir , 'face.jpg'), face_frame)



    #         # PIL Image crop and save

    #         img1 = Image.open(directory + '/face.jpg')
    #         img2 = Image.open(directory + '/face.jpg')

    #         # img1.crop(0, 0, int(40 * WIDTH // 100), HEIGHT)
    #         # img2.crop(int(40 * WIDTH // 100), 0, WIDTH, HEIGHT)

    #         card_area = (0, 0, int(40 * WIDTH // 100), HEIGHT)
    #         face_area = (int(40 * WIDTH // 100), 0, WIDTH, HEIGHT)
    #         img1 = img1.crop(card_area)
    #         img2 = img2.crop(face_area)

    #         img1.save('./card/card.jpg')
    #         img2.save('./face/face.jpg')

    #         full_file_path = './face/face.jpg'
    #         predictions = predict(full_file_path, img, model_path="trained_knn_model.clf")
    #         names = []
    #         try:
    #             for name, (top, right, bottom, left) in predictions:
    #                 names.append(name)
    #         except:
    #             pass
    #         if len(resp) == 0:
    #             names= ["Face not detected"]

    #             resp = names[0]
    #     except:
    #         pass


    # return resp


# def predict(X_img_path,img , knn_clf=None, model_path=None, distance_threshold=0.48):

#     # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
#     #     raise Exception("Invalid image path: {}".format(X_img_path))

#     # if knn_clf is None and model_path is None:
#     #     raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

#     # Load a trained KNN model (if one was passed in)
#     if knn_clf is None:
#         with open(model_path, 'rb') as f:
#             knn_clf = pickle.load(f)


#     # Load image file and find face locations
#     # X_img = face_recognition.load_image_file(X_img_path)
#     X_img = img
#     X_face_locations = face_recognition.face_locations(X_img)

#     # If no faces are found in the image, return an empty result.
#     if len(X_face_locations) == 0:
#         return []

#     # Find encodings for faces in the test iamge
#     faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

#     # Use the KNN model to find the best matches for the test face
#     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

#     # Predict classes and remove classifications that aren't within the threshold
#     return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def predict(X_img_path, faces,  knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
        of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations

    X_img = faces
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # print(closest_distances[0][0][0])
    are_matches = [closest_distances[0][i][0] <=
        distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

if __name__ == '__main__':
	 app.run(debug=True)