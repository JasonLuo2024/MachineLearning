__author__ = "JasonLuo"
from tqdm import tqdm
from PIL import Image
import os





directory = r'D:/Eastern_Health_Age'
for category in tqdm(['Normal', 'Abnormal']):
    path = os.path.join(directory, category)
    left_path = os.path.join(path, 'MLO')
    right_path = os.path.join(path, 'CC')
    for LR in [right_path, left_path]:
        for root, directories, files in os.walk(LR):
           for file in files:
               file_path = os.path.join(root, file)
               try:
                   Image.open(file_path).convert('RGB')
               except OSError as e:
                   print("file path removed:", file_path)
                   os.remove(file_path)


