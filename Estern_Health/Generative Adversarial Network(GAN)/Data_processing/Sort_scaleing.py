__author__ = "JasonLuo"
import pydicom
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ImageOps
from skimage import measure
import cv2

## this file is used to sort DCM file based on it viewPosition - CC and MLO
## and convert all the DCM images into PNG images
## then using LOG filter to upscale one image to six images

directory = r'C:\Users\Woody\Desktop\fake_PNG'
target = r'C:\Users\Woody\Desktop\Fake_scaling'

def multi_scale_analysis(image_path):
    image = cv2.imread(image_path)
    image = np.array(image)
    scales = [ 5 ,7, 9,11,13 , 15]
    results = []
    for scale in scales:
        # Apply Laplacian operator with current scale
        filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        results.append(filtered)
    return results

for category in tqdm(['Normal', 'Abnormal']):
    path = os.path.join(directory, category)
    target_path = os.path.join(target, category)
    os.makedirs(path, exist_ok=True)
    os.makedirs(target_path , exist_ok=True)
    for save_class in tqdm(['MLO', 'CC']):
        filePath = os.path.join(path, save_class)
        os.makedirs(filePath, exist_ok=True)
        save_path = os.path.join(target_path, save_class)
        os.makedirs(save_path, exist_ok=True)
        invalid_count = 0
        for subpath in tqdm(os.listdir(path)):
            dicom_files = [os.path.join(filePath, filename) for filename in os.listdir(filePath) if
                           filename.endswith('.png')]
            for file_path in dicom_files:
                try:
                    png_Image = multi_scale_analysis(file_path)
                    base_name = os.path.splitext(subpath)[0]
                    for i, result in enumerate(png_Image):
                        output_filename = f"{base_name}_scale{i + 1}.png"
                        if os.path.exists(os.path.join(save_path, output_filename)):
                            unique_id = 1
                            while os.path.exists(os.path.join(save_path, output_filename)):
                                output_filename = f"{base_name}_scale{i + 1}_{unique_id}.png"
                                unique_id += 1
                        output_path = os.path.join(save_path, output_filename)
                        cv2.imwrite(output_path, result)
                except KeyError:
                    invalid_count = invalid_count + 1
        print(invalid_count)





