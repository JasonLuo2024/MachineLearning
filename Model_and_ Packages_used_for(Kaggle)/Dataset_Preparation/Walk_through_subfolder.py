import pydicom
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
from skimage import measure
import cv2
import pandas as pd
import threading
import concurrent.futures
dirctory = r'C:\Users\Woody\Desktop\RSNA_PNG'
train_csv = r'C:\Users\Woody\Desktop\RSNA_PNG\train.csv'

def multi_scale_analysis(image):
    image = np.array(image)
    scales = [ 5 ,7, 9,11,13 , 15]
    results = []
    for scale in scales:
        # Apply Laplacian operator with current scale
        filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        results.append(filtered)
    return results

for label, category in enumerate(['Normal', 'Abnormal']):
   target_path = os.path.join(dirctory, category)
   os.makedirs(target_path , exist_ok=True)
   CC_path = os.path.join(target_path, 'CC')
   MLO_path = os.path.join(target_path, 'MLO')

   for folder_path in [CC_path, MLO_path]:
       for root, directories, files in tqdm(os.walk(folder_path)):
           for file in files:
               file_path = os.path.join(root, file)
               try:
                   image = cv2.imread(file_path)
                   png_Image = multi_scale_analysis(image)
                   for i, result in enumerate(png_Image):
                       output_filename = f"{file}.png"
                       if os.path.exists(os.path.join(root, output_filename)):
                           unique_id = 1
                           while os.path.exists(os.path.join(root, output_filename)):
                               output_filename = f"{file}_scale{i + 1}_{unique_id}.png"
                               unique_id += 1
                       output_path = os.path.join(root, output_filename)
                       cv2.imwrite(output_path, result)
               except Exception as e:
                   print("Error: An exception occurred while processing the DICOM file.")
                   print(e)

