import time
import numpy as np
import pandas as pd
import pickle
import os
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
import json

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

def main():
    credentials = service_account.Credentials.from_service_account_file("./sylvan-airship-406701-7556ece80fd0.json")
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket('full_images_2024')

    directory = './celebrity_info'
    with open('./stratified_sample_names.pkl', 'rb') as f:
        celebrity_names = pickle.load(f)
        
    for name in celebrity_names:
        uuid_dict = {}
        print(f'Now downloading: {name}')
        success_download = 0
        df = pd.read_pickle(f"{directory}/{name}.pkl")
        df['downloaded'] = 0
        for index, row in df.iterrows():
            img_link = row['image_link']          
            uuid = row['uuid']                      
            img_name = row['outlet'] + "_" + uuid
                
            try:
                response = requests.get(img_link, timeout=10)
            except Exception as e:
                continue
                    
            if response.status_code != 200:
                url = get_wayback_url(img_link)
                if url != None:
                    try:
                        response = requests.get(img_link, timeout=10)
                    except:
                        continue
                    if response.status_code != 200:
                        continue
                else:
                    continue
    
            try:
                content = BytesIO(response.content)
                image = Image.open(content)
                img_format = image.format.lower()       
                image = image.convert('RGB')   
                width, height = image.size
                np_image = np.array(image)  
                
                content.seek(0)
                image_blob = bucket.blob(f"celebrity_images2/{name}/{img_name}.{img_format}")
                image_blob.upload_from_file(content)
                
                df.at[index, 'downloaded'] = 1
                success_download += 1

                content.seek(0)
                image_blob = bucket.blob(f"benchmark_images/{name}/{name}_{success_download}.{img_format}")
                image_blob.upload_from_file(content)

                uuid_dict[f'{name}_{success_download}'] = img_name
                
                del image
                gc.collect()
                
                if success_download == 50:
                    break

            except:
                continue

        with open(f'./benchmark_uuid/{name}_uuid.pkl', 'wb') as f:
            pickle.dump(uuid_dict, f)
        df = df[df['downloaded'] == 1]
        df.to_csv(f'./celebrity_downloaded_info/{name}.csv', index=False)
        print(f'Info of {name} downloaded to local')

if __name__ == '__main__':
    main()
