import pydicom
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ImageOps
from skimage import measure
import cv2
import pandas as pd
import threading
import concurrent.futures
import shutil
dirctory = r'C:\Users\Woody\Desktop\RSNA_PNG'
train_csv = r'C:\Users\Woody\Desktop\RSNA_PNG\train.csv'

for category in ['Normal', 'Abnormal']:
   target_path = os.path.join(dirctory, category)
   os.makedirs(target_path , exist_ok=True)
   CC_path = os.path.join(target_path, 'CC')
   MLO_path = os.path.join(target_path, 'MLO')
   os.makedirs(CC_path, exist_ok=True)
   os.makedirs(MLO_path, exist_ok=True)
   for filePath in [CC_path,MLO_path]:
       dicom_files = [os.path.join(filePath, filename) for filename in os.listdir(filePath) if filename.endswith('.png')]
       for PNG_file in tqdm(dicom_files):
           try:
               image_id = int(os.path.basename(PNG_file).split(".")[0])
               df = pd.read_csv(train_csv, delimiter=',')
               # Filter the DataFrame based on the patient's ID
               filtered_df = df[df['image_id'] == image_id]
               if not filtered_df.empty:
                   patient_age = str(int(filtered_df.iloc[0]['age']))
                   save_path = os.path.join(filePath, patient_age)
                   os.makedirs(save_path, exist_ok=True)
                   shutil.move(PNG_file, save_path)
           except Exception as e:
               print(e)

