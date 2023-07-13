import os

from tqdm.auto import tqdm
num_dcm_files = 0
directory = r'D:\WholeDataSet'
for category in tqdm(['Normal', 'Abnormal']):
    path = os.path.join(directory, category)
    for subpath in tqdm(os.listdir(path)):
        filePath = os.path.join(path, subpath)
        for file_name in tqdm(os.listdir(filePath)):
            if file_name.endswith('.dcm'):  # Check if the file is a DICOM file
                num_dcm_files += 1



print(f"The folder contains {num_dcm_files} DICOM files.")
