import time
import numpy as np
import pandas as pd
import pickle
import cv2
from deepface import DeepFace
import os
from retinaface import RetinaFace
from imutils import build_montages
from PIL import Image
from io import BytesIO
import io
import gc
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import storage
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
from datetime import datetime
import argparse
import random


def main(args):
    credentials = service_account.Credentials.from_service_account_file("./sylvan-airship-XXX.json")
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket('full_images_2024')
    with open('./stratified_sample_names.pkl', 'rb') as f:
        names = pickle.load(f)


    # model = args.model
    backend = args.backend

    models = ['Facenet512', 'Facenet', 'VGG-Face']
    # backends = ['yolov8', 'yunet']
    alignment_modes = [True, False]

    for name in names:
        embeddings_dict_noalign1 = {}
        embeddings_dict_noalign2 = {}
        embeddings_dict_noalign3 = {}
        embeddings_dict_align1 = {}
        embeddings_dict_align2 = {}
        embeddings_dict_align3 = {}
        # uuid_dict = {} # {'Trump_1': '', 'Trump2': ''}

        blobs = bucket.list_blobs(prefix=f'benchmark_images/{name}')
        blobs = list(blobs)
        print(name)

        for blob in blobs:
            img_name = blob.name[blob.name.rfind('/')+1:blob.name.rfind('.')]

            image_data = blob.download_as_bytes()
            with Image.open(BytesIO(image_data)) as image:
                width, height = image.size
                img_format = image.format.lower()
                np_image = np.array(image)
                byte_io = BytesIO()

                for model in models:
                    for alignment_mode in alignment_modes:
                        try:
                            face_embeddings = DeepFace.represent(
                                img_path = np_image,
                                model_name = model,
                                detector_backend = backend,
                                align = alignment_mode,
                                enforce_detection = True,
                            )
                        except Exception as e:
                            print(e)
                            continue

                        n_face = 1
                        for embed in face_embeddings:
                            face_name = f'{img_name}_{n_face}'
                            if alignment_mode:
                                if model == 'Facenet':
                                    embeddings_dict_align1[face_name] = embed
                                elif model == 'Facenet512':
                                    embeddings_dict_align2[face_name] = embed
                                else:
                                    embeddings_dict_align3[face_name] = embed
                            else:
                                if model == 'Facenet':
                                    embeddings_dict_noalign1[face_name] = embed
                                elif model == 'Facenet512':
                                    embeddings_dict_noalign2[face_name] = embed
                                else:
                                    embeddings_dict_noalign3[face_name] = embed

                            if alignment_mode and model == 'Facenet':
                                x, y, w, h = embed['facial_area']['x'], embed['facial_area']['y'], embed['facial_area']['w'], embed['facial_area']['h']
                                face = image.crop((max(0, x-0.05*w), max(0,y-0.05*h), min(x + w*1.05, width), min(y + 1.05*h, height)))
                                output_stream = BytesIO()
                                face.save(output_stream, format=img_format)
                                output_stream.seek(0)
                                new_blob = bucket.blob(f"benchmark_faces/{backend}/{name}/{face_name}.{img_format}")
                                new_blob.upload_from_file(output_stream)

                            n_face += 1

                        del face_embeddings
                        gc.collect()

        with open(f'./embeddings_align/{backend}/Facenet/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_align1, file)
        with open(f'./embeddings_align/{backend}/Facenet512/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_align2, file)
        with open(f'./embeddings_align/{backend}/VGG-Face/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_align3, file)

        with open(f'./embeddings_noalign/{backend}/Facenet/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_noalign1, file)
        with open(f'./embeddings_noalign/{backend}/Facenet512/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_noalign2, file)
        with open(f'./embeddings_noalign/{backend}/VGG-Face/{name}.pkl', 'wb') as file:
            pickle.dump(embeddings_dict_noalign3, file)


        del image, image_data
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloading Images')
    parser.add_argument('-i', '--idx', type=int, default = -1,
                    help='index')
    # parser.add_argument('-m', '--model', type=str, default = '',
    #                 help='model')
    parser.add_argument('-b', '--backend', type=str, default = '',
                    help='backend')
    args = parser.parse_args()
    main(args)
                                                                        
                                                                                                                                                                                          