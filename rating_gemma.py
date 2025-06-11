import google.generativeai as genai
import pandas as pd
import time
from google.generativeai import GenerationConfig
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import io
import ast
import os

def draw_bounding_box_from_url(image_url, bbox_coords, box_color='red', box_width=3):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except:
        try: # try requesting again
            response = requests.get(image_url)
            response.raise_for_status()
        except:
            print(f"Error fetching image from {image_url}")
            return None
    
    image = Image.open(io.BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    x1, y1, w, h = bbox_coords
    draw.rectangle([x1, y1, x1+w, y1+h], outline=box_color, width=box_width)
    return image

def main(API_KEY):
    # import data
    df = pd.read_csv('refined_sample_mutated.csv')
    df['coord'] = df['coord'].apply(ast.literal_eval) 

    # set up model
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemma-3-27b-it")

    error_idx = []
    n = len(df)
    print(f"Total images to process: {n}")

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f'Now processing {i}')
        if i % 1000 == 0:
            df.to_csv(f'refined_sample_ratings_{i}.csv', index=False)
            try:
                os.remove(f'refined_sample_ratings_{i-1000}.csv')
            except:
                print("File not found")

        person = row['person']

        prompt = f"""
            How favorably does this image portray {person} in the red box? Rate it on a 3-point scale, where:
            - 1 means unfavorable,
            - 2 means a neutral midpoint,
            - 3 means favorable.
            Begin your response with the rating number, followed by a short explanation of your reasoning.
            Please separate the rating number and explanation with ";". Do not use semicolons anywhere else.
            """

        imgurl = row['gcs_url']
        coords = row['coord']

        try:
            boxed_image = draw_bounding_box_from_url(imgurl, bbox_coords=coords)

            img_bytes_io = BytesIO()
            boxed_image.save(img_bytes_io, format='JPEG') 

            img_bytes_io.seek(0)
            img_bytes = img_bytes_io.read()

            # generate response
            response = model.generate_content([
                    {"text": prompt},
                    {"inline_data": {"mime_type": f"image/jpeg", "data": img_bytes}}
                ],
                generation_config=GenerationConfig(temperature=0)
            )
            
            if response.candidates and response.candidates[0].content.parts: # check if valid response
                res_parts = response.text.strip().split(sep=";")
                res_parts = [x.strip() for x in res_parts]
                if len(res_parts) != 2:
                    rating = response.text[0]
                    explanation = response.text[1:].strip()

                    if not rating.isnumeric():
                        error_idx.append(i)
                        print(f"error getting rating for {imgurl}")
                        continue
                    df.at[i, 'rating'] = int(rating)
                    df.at[i, 'explanation'] = explanation
                    continue
                rating, exp = res_parts[0], res_parts[1]

                if not rating.isnumeric():
                    print(f"error getting rating for {imgurl}")
                    continue

                rating = int(rating)

                df.at[i, 'rating'] = rating
                df.at[i, 'explanation'] = exp

            else:
                error_idx.append(i)
                print(f"No valid content returned for image {i}. Finish reason:", response.candidates[0].finish_reason)

        except Exception as e:
            print(f"Error processing image {i}: {e}")
            error_idx.append(i)

    df.to_csv('refined_sample_ratings.csv', index=False)

    return error_idx


if __name__ == '__main__':
    API_KEY = ""  # Fill in your Google API key here
    error_idx = main(API_KEY)
    print(f"Error IDX: {error_idx}")