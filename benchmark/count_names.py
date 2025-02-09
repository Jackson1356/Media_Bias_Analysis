import time
import numpy as np
import pandas as pd
import pickle
import os
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
from io import StringIO
import json
import re

def main():
    credentials = service_account.Credentials.from_service_account_file("./sylvan-airship-XXX.json")
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket('full_images_2024')

    with open('./stratified_sample_names.pkl', 'rb') as f:
        celebrity_names = pickle.load(f)
    
    name_count = {celebrity: 0 for celebrity in celebrity_names}
    years = ['2017', '2018', '2019', '2020', '2021']
    outlets = ['USAtoday', 'abc', 'alternet', 'atlantic', 'breitbart', 'business', 'buzz', 'cbs', 
               'cnn', 'dailybeast', 'dailycaller', 'dailywire', 'epochtimes', 'federalist', 'foxnews', 'freebeacon','gatewaypundit',
               'guardian', 'huff', 'info', 'intercept', 'motherjones',
              'msnbc', 'nationalreview', 'nbc', 'newsbusters', 'newsmax', 'newsweek'
              'npr', 'nypost', 'nyt', 'oann', 'pbs', 'pjmedia', 'politicalinsider',
              'politico', 'realclearpolitics', 'redstate', 'rightscoop', 'salon', 'slate',
              'spectator', 'theblaze', 'thehill', 'townhall', 'USAtoday', 'vox', 'waexam',
              'wapost', 'watimes', 'wj', 'wsj']

    for outlet in outlets:
        df = pd.DataFrame(columns=['uuid', 'headline', 'outlet', 'image_link', 'celebrity'])
        for year in years:
            file_name = outlet + year
            print(file_name)
            bucket = client.bucket('full_images_2024')
            
            blob = bucket.blob(f'data/{file_name}.pkl')

            try: 
                pickle_data = blob.download_as_bytes()
            except:
                print('file not found!!!')
                continue
                
            data_io = BytesIO(pickle_data)
            articles = pd.read_pickle(data_io)
            del data_io
            gc.collect()
            
            col = ['uuid', 'headline', 'image_link'] 
            articles = articles[col]
            articles['outlet'] = file_name
            articles['celebrity'] = articles.apply(lambda x: [], axis=1)
            
            for _, entry in articles.iterrows():
                for name in celebrity_names:
                    if re.search(r'\b' + re.escape(name) + r'\b', entry['headline']):
                        entry['celebrity'].append(name)
                        name_count[name] += 1
                        
            
            df = pd.concat([df, articles], ignore_index=True)
            
            del articles
            gc.collect()
        df.to_pickle(f'./name_count/{outlet}_names.pkl')
        with open(f'./name_count/{outlet}_name_count.json', 'w') as file:
            json.dump(name_count, file)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Downloading Images')
    # parser.add_argument('-n', '--outlet', type=str,
    #                 help='name of the outlet')
    # args = parser.parse_args()
    main()
