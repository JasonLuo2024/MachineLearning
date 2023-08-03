__author__ = "JasonLuo"
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
def multi_scale_analysis(image):
    image = np.array(image)
    scales = [ 5 ,7, 9,11,13 , 15]
    results = []
    for scale in scales:
        # Apply Laplacian operator with current scale
        filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        results.append(filtered)
    return results
def dicom_to_png(dicom_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Get the pixel data
    pixel_array = ds.pixel_array

    # Normalize the pixel values
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_range = pixel_max - pixel_min

    normalized_array = (pixel_array - pixel_min) / pixel_range * 255.0

    # Convert the pixel array to a PIL image
    image = Image.fromarray(normalized_array.astype('uint8'))
    image_array = np.array(image)
    return image_array



def crop_breast_region(image):
    # Convert the image to grayscale numpy array
    image_array = np.array(image)

    # Apply Otsu's thresholding to create a binary image
    _, binary_image = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming it represents the breast)
    breast_contour = max(contours, key=cv2.contourArea)

    # Create a convex hull around the breast contour
    convex_hull = cv2.convexHull(breast_contour)

    # Create a mask for the convex hull
    mask = np.zeros_like(image_array)
    cv2.drawContours(mask, [convex_hull], 0, 255, -1)

    # Apply the mask to the original image
    breast_image = cv2.bitwise_and(image_array, mask)

    # Find the bounding box of the convex hull
    x, y, w, h = cv2.boundingRect(convex_hull)

    # Crop the breast region from the original image
    cropped_image = image_array[y:y + h, x:x + w]

    return cropped_image


# Specify the directory containing the DICOM files

train_csv = r'D:\RSNA\train.csv'
test_csv = r'D:\RSNA\test.csv'
directory = r'D:\RSNA'
target = r'D:\RSNA_PNG'

def process_dcm (directory,subpath,train_csv,target):
    filePath = os.path.join(directory, subpath)
    dicom_files = [os.path.join(filePath, filename) for filename in os.listdir(filePath) if filename.endswith('.dcm')]
    for filename in dicom_files:
        file_name = os.path.basename(filename)
        file_number = os.path.splitext(file_name)[0]

        image_id = int(file_number.split('\\')[-1])
            # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(train_csv, delimiter=',')

            # Filter the DataFrame based on the patient's ID
        filtered_df = df[df['image_id'] == image_id]

            # Check if any rows match the patient ID
        if not filtered_df.empty:
            cancer_value = filtered_df.iloc[0]['cancer']
            viewposition= filtered_df.iloc[0]['view']
            class_path = os.path.join(target,"Normal") if cancer_value == 0 else os.path.join(target,"Abnormal")
            save_path = os.path.join(class_path,"MLO") if viewposition == "MLO" else os.path.join(class_path,"CC")
            try:
                png_Image = dicom_to_png(filename)
                base_name = image_id
                # for i, result in enumerate(png_Image):
                output_filename = f"{base_name}.png"
                # if os.path.exists(os.path.join(save_path, output_filename)):
                #     unique_id = 1
                #     while os.path.exists(os.path.join(save_path, output_filename)):
                #         output_filename = f"{base_name}_scale{i + 1}_{unique_id}.png"
                #         unique_id += 1
                output_path = os.path.join(save_path, output_filename)
                cv2.imwrite(output_path, png_Image)
            except Exception as e:
                print("Error: An exception occurred while processing the DICOM file.")
                print(e)


for category in ['Normal', 'Abnormal']:
   target_path = os.path.join(target, category)
   os.makedirs(target_path , exist_ok=True)
   CC_path = os.path.join(target_path, 'CC')
   MLO_path = os.path.join(target_path, 'MLO')
   os.makedirs(CC_path, exist_ok=True)
   os.makedirs(MLO_path, exist_ok=True)

num_threads = 12

# Create a list to store the DICOM subdirectories
dicom_subdirectories = []

# Traverse the directory and collect DICOM subdirectories
for subpath, dirs, files in tqdm(os.walk(directory)):
    # Check if the current subdirectory contains DICOM files
    if any(file.endswith('.dcm') for file in files):
        dicom_subdirectories.append(subpath)

# Calculate the actual number of threads based on the available DICOM subdirectories and desired number of threads
num_actual_threads = min(len(dicom_subdirectories), num_threads)

# Create a ThreadPoolExecutor with the desired number of threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_actual_threads) as executor:
    # Submit the process_dcm function to the executor for each DICOM subdirectory
    futures = []
    for subpath in dicom_subdirectories:
        future = executor.submit(process_dcm, directory, subpath, train_csv, target)
        futures.append(future)

    # Create a progress bar with the total number of DICOM files
    progress_bar = tqdm(total=len(futures), desc='Processing DICOM files', unit='file')

    # Wait for all futures (threads) to complete
    for future in concurrent.futures.as_completed(futures):
        # Handle any exceptions that occurred in the threads, if needed
        try:
            future.result()
        except Exception as e:
            print("An exception occurred:", str(e))

        # Update the progress bar for each completed future (processed DICOM file)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()