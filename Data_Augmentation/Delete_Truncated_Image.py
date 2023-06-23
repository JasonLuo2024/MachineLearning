__author__ = "JasonLuo"
from tqdm import tqdm
from PIL import Image
import os


def find_truncated_images(image_list):
    truncated_images = []

    for image_file in image_list:
        try:
            Image.open(image_file).verify()
        except OSError as e:
            if str(e) == "image file is truncated":
                truncated_images.append(image_file)

    return truncated_images



directory = r'D:/WholeDataTest2'
truncated_images = []
for category in tqdm(['Normal', 'Abnormal']):
    path = os.path.join(directory, category)
    left_path = os.path.join(path, 'MLO')
    right_path = os.path.join(path, 'CC')
    for LR in [right_path, left_path]:
        png_files = [os.path.join(LR, filename) for filename in os.listdir(LR) if filename.endswith('.png')]
        for file_path in tqdm(png_files):
            try:
                Image.open(file_path).verify()
            except OSError as e:
                print("file path removed:", file_path)
                os.remove(file_path)

