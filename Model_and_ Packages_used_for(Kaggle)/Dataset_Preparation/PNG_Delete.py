__author__ = "JasonLuo"
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
from skimage import measure
import pandas as pd
import threading
import concurrent.futures
dirctory = r'/home/hluo/RSNA_PNG'


for label, category in enumerate(['Normal', 'Abnormal']):
   target_path = os.path.join(dirctory, category)
   os.makedirs(target_path , exist_ok=True)
   CC_path = os.path.join(target_path, 'CC')
   MLO_path = os.path.join(target_path, 'MLO')

   for folder_path in [CC_path, MLO_path]:
       for root, directories, files in tqdm(os.walk(folder_path)):
           for file in files:
               if file.endswith(".png") and not file .endswith(".png.png") and not file .endswith(".png_scale1_1.png") and not file .endswith(".png_scale2_1.png")and not file .endswith(".png_scale3_1.png")and not file .endswith(".png_scale4_1.png")and not file .endswith(".png_scale5_1.png") and not file .endswith(".png_scale6_1.png") :
                   os.remove(os.path.join(root, file))

