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
    credentials = service_account.Credentials.from_service_account_file("./sylvan-airship-406701-7556ece80fd0.json")
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket('full_images_2024')

    # with open('name_tokens.pkl', 'rb') as file:
    #     celebrities = pickle.load(file)
    # celebrity_sets = {celeb: set(names) for celeb, names in celebrities.items()}

    celebrity_names = ['Young Kim', 'Ozuna', 'Bob Corker', 'Tony Romo', 'Jeanine Pirro', 'Tom Perez', 'Vanessa Bryant', 'Van Jones', 
                       'Russell Wilson', 'Jack Ma', 'Shohei Ohtani', 'Aung San Suu Kyi', 'George Soros', 'Ralph Northam', 'Bubba Wallace', 
                       'Carmelo Anthony', 'David Ortiz', 'Whoopi Goldberg', 'Lewis Hamilton', 'Lil Nas X', 'Masayoshi Son', 'Clarence Thomas', 
                       'Katie Hill', 'Dak Prescott', 'Priyanka Chopra', 'Oprah Winfrey', 'Urban Meyer', 'Patrick Mahomes', 'Marco Rubio', 
                       'Ro James', 'Lin-Manuel Miranda', 'Bryce Harper', 'Rihanna', 'Naomi Osaka', 'Roseanne Barr', 'Kayleigh McEnany', 
                       'Andrew Yang', 'Satya Nadella', 'Conor McGregor', 'Nicki Minaj', 'Zach Braff', 'Kevin Durant', 'Alex Rodriguez', 
                       'Nadia Murad', 'Trevor Noah', 'George Foreman', 'Ruth Bader Ginsburg', 'Neymar', 'Sam Elliott', 'Seema Verma', 
                       'Robert Menendez', 'Jim Acosta', 'Nicole Malliotakis', 'Susan Rice', 'Rashida Tlaib', 'Ma Jun', 'Anthony Gonzalez', 
                       'Chrissy Teigen', 'Venus Williams', 'Camilo', 'Jennifer Aniston', 'Lil Pump', 'Caitlyn Jenner', 'Xi Jinping', 
                       'Demi Lovato', 'Rosie Perez', 'J. Cole', 'Mitch Landrieu', 'Mike Garcia', 'Maya Rudolph', 'Kevin Cramer', 'Master P', 
                       'Chloé Zhao', 'Henry Cuellar', 'Rachel Weisz', 'Linda Ronstadt', 'Wanda Sykes', 'Luis Fonsi', 'Breonna Taylor', 
                       'Tayshia Adams', 'Smokey Robinson', 'Andy Kim', 'Ruth B', 'Fareed Zakaria', 'Darren Criss', 'Angela Davis', 
                       'Omar al-Bashir', 'Jesse Williams', 'Mariah Carey', 'Yalitza Aparicio', 'Anya Taylor-Joy', 'Jodie Foster', 
                       'Colin Kaepernick', 'Cardi B', 'Tan France', 'Tessa Thompson', 'Selena Gomez', 'Leana Wen', 'Oscar Isaac', 
                       'Carrie Ann Inaba', 'Malcolm Gladwell', 'Simone Biles', 'Tiger Woods']
    
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
