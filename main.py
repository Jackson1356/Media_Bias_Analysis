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

def get_wayback_url(image_url):
    # Access the Wayback Machine API to get the closest archive snapshot
    wayback_api_url = f"https://archive.org/wayback/available?url={image_url}"
    response = requests.get(wayback_api_url)
    data = response.json()

    if 'archived_snapshots' in data and data['archived_snapshots']:
        closest_snapshot = data['archived_snapshots']['closest']
        if closest_snapshot['available']:
            wayback_url = closest_snapshot['url']
            return wayback_url
    return None

def main(args):
    credentials = service_account.Credentials.from_service_account_file(
    "./sylvan-airship-406701-7556ece80fd0.json")
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket('full_images_2024')
    
    file_name = args.file_name
    blob = bucket.blob(f'data/{file_name}.pkl')
    pickle_data = blob.download_as_bytes()
    data_io = BytesIO(pickle_data)
    articles = pd.read_pickle(data_io)
    alignments_data = {} #key: name; value: outlet, date, uuid, embedding, confidence_level, facial_area
    col = ['uuid', 'publication_timestamp', 'link', 'headline', 'excerpt', 'source', 'image_link', 'article', 'fb_engage'] 
    articles = articles[col]
    articles['source'] = articles['source'].apply(lambda x: x['publisher'])
    articles['date'] = articles['publication_timestamp']
    articles.drop('publication_timestamp', axis=1, inplace=True)
    articles['n-faces'] = 0
    articles['status'] = 0

    if args.start_idx != -1 and args.end_idx != -1:
        sliced = True
        print(f'File: {file_name}, from {args.start_idx} to {args.end_idx}, total length: {len(articles.index)}')
        articles = articles[args.start_idx:args.end_idx]
    elif args.start_idx != -1 and args.end_idx == -1: 
        sliced = True
        print(f'File: {file_name}, from {args.start_idx} to {len(articles.index)}, total length: {len(articles.index)}')
        articles = articles[args.start_idx:len(articles.index)]
    else:
        sliced = False
        print(f'File: {file_name} entire, total length: {len(articles.index)}')

    blobs = bucket.list_blobs()
    model = args.model

    t1 = time.time()

    for index, row in articles.iterrows():
        if index % 100 == 0:
            print(f'Currently Downloading No. {index}')
        img_link = row['image_link']          
        uuid = row['uuid']                      
        img_name = file_name + "_" + uuid
        
        if type(img_link) is not str:
            articles.at[index, 'status'] = 0
    
        if type(uuid) is not str:
            articles.at[index, 'status'] = 0
            
        try:
            try:
                response = requests.get(img_link, timeout=10)
            except Exception as e:
                print(f'Request ERROR: {e}')
                articles.at[index, 'status'] = 0
                continue
                    
            if response.status_code != 200:
                url = get_wayback_url(img_link)
                if url != None:
                    try:
                        response = requests.get(img_link, timeout=4)
                    except:
                        articles.at[index, 'status'] = 0
                    if response.status_code != 200:
                        articles.at[index, 'status'] = response.status_code
                        continue
                else:
                    articles.at[index, 'status'] = response.status_code
                    continue
        
        
            outlet = img_name.split("_")[0]
            content = BytesIO(response.content)
            image = Image.open(content)
            img_format = image.format.lower()       
            image = image.convert('RGB')   
            width, height = image.size
            np_image = np.array(image)  
        
            # Generate alignmnets and upload alignments and image
            face_embeddings = DeepFace.represent(
                    img_path = np_image,
                    model_name = model,
                    detector_backend = 'yolov8', 
                    align = False,
                    enforce_detection = True,
            )
    
            articles.at[index, 'n-faces'] = len(face_embeddings)       
            n = 1
            for embed in face_embeddings:
    #            if embed['face_confidence'] < 0.74:
    #                continue
                x, y, w, h = embed['facial_area']['x'], embed['facial_area']['y'], embed['facial_area']['w'], embed['facial_area']['h']
                # add padding to the alignment
                cropped_image = image.crop((max(0, x-0.05*w), max(0,y-0.05*h), min(x + w*1.05, width), min(y + 1.05*h, height))) 
                cropped_img_name = f'{img_name}_alignment_{n}'
                n += 1
    
                alignments_data[cropped_img_name] = {}
                alignments_data[cropped_img_name]['outlet'] = outlet[:-4]
                alignments_data[cropped_img_name]['date'] = row['date']
                alignments_data[cropped_img_name]['uuid'] = uuid
                alignments_data[cropped_img_name]['embedding'] = embed['embedding']
                alignments_data[cropped_img_name]['confidence_level'] = embed['face_confidence']
                alignments_data[cropped_img_name]['facial_area'] = embed['facial_area']
		alignments_data[cropped_img_name]['name'] = cropped_img_name
        
                output_stream = BytesIO()
                cropped_image.save(output_stream, format=img_format)
                output_stream.seek(0)
                new_blob = bucket.blob(f"alignments/{outlet[:-4]}/{outlet}/{cropped_img_name}.{img_format}")
                new_blob.upload_from_file(output_stream)
    #            print(f'     Alignment Uploaded: {cropped_img_name}')
        
            content.seek(0)
            blob = bucket.blob(f"images/{outlet[:-4]}/{outlet}/{img_name}.{img_format}")
            blob.upload_from_file(content)
    #        print(f"Image uploaded: {img_name}")
            articles.at[index, 'status'] = 200
            if n > 1:
                del image, cropped_image, face_embeddings
            else:
                del image, face_embeddings
            gc.collect()
    
        except ValueError as ve:
            articles.at[index, 'status'] = 100  # no face
    
        except Exception as e:
            print(f'E: {e}')
            articles.at[index, 'status'] = 0

    t2 = time.time()
    print(f'Finished downloading! Total aligments: {len(alignments_data)}')
    print(f'Total time: {t2-t1}')
    
    alignment_file_name = file_name
    outlet_file_name = f'{file_name}_info'
    if sliced:
        alignment_file_name = f'{file_name}_{args.end_idx}'
        outlet_file_name = f'{file_name}_{args.end_idx}_info'
    
    alignment_data_df = pd.DataFrame(alignments_data)
    alignment_data_df = alignment_data_df.transpose()
    try:
        alignment_info_buffer = io.StringIO()
        alignment_data_df.to_json(alignment_info_buffer, orient='records', lines=True)
        alignment_info_buffer.seek(0)
        
        blob = bucket.blob(f'alignment_info/{alignment_file_name}.json')
        blob.upload_from_file(alignment_info_buffer)
        print('Alignment info downloaded to google bucket')   
    except Exception as e:
        print(e)
        alignment_data_df.to_json(f'{alignment_file_name}.json', orient='records', lines=True)
        print('Alignment info downloaded to local')    

    try:
        articles_info_buffer = io.StringIO()
        articles.to_csv(articles_info_buffer, index=False)
        articles_info_buffer.seek(0)
        
        blob = bucket.blob(f'outlet_info/{outlet_file_name}.csv')
        blob.upload_from_file(articles_info_buffer)
        print('Outlet info downloaded to google bucket')
    except Exception as e:
        print(e)
        articles.to_csv(f'{outlet_file_name}.csv', index=False)
        print('Outlet info downloaded to local')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloading Images')
    parser.add_argument('-n', '--file_name', type=str,
                    help='name of the outlet')
    parser.add_argument('-s', '--start_idx', type=int, default = -1,
                    help='start index')
    parser.add_argument('-e', '--end_idx', type=int, default = -1,
                    help='end index')
    parser.add_argument('-m', '--model', default='Facenet', type=str,
                    help='The model used to generate embeddings')
    args = parser.parse_args()
    main(args)