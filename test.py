import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import urllib3
import urllib.request
import os
import cv2
from tempfile import NamedTemporaryFile
from google.cloud import storage

def get_dataframes():

    jsonSeries = pd.read_json('gs://foodygs/foodyai_data/Training_2/annotations.json',typ='series')
    # jsonseries has three dataframe and it could be retrieved as the following

    categories = pd.DataFrame(jsonSeries.categories)  
    print(f'categories information containing {len(categories)} rows has been received')
    
    images = pd.DataFrame(jsonSeries.images)
    print(f'image information containing {len(images)} rows has been received')

    annotations = pd.DataFrame(jsonSeries.annotations)    
    print(f'annotations information containing {len(annotations)} rows has been received')

    nutrition = pd.read_csv('gs://foodygs/Nutrition/nutrition.csv', sep=",", index_col=0)
    print(f'nutrition information containing {len(nutrition)} has been received')


    
    
    return categories,images,annotations,nutrition

get_dataframes()

def get_an_image(img_url_and_name='https://storage.cloud.google.com/foodygs/foodyai_data/Training_2/images/006316.jpg'):
    # resp = urllib.request.urlopen(img_url_and_name)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.imread(img_url_and_name, 0)
    print(type(image))
    return image

# get_an_image('https://storage.cloud.google.com/foodygs/foodyai_data/Training_2/images/006316.jpg')