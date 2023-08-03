import pydicom
import os
from shutil import copyfile
from tqdm.auto import tqdm


directory = 'D:/WholeDataSet'
destination = 'D:/WholeDataSet/data_withoutCC&MLO'
invalid_count = 0
actual_iamge = 0
for category in tqdm(['Normal', 'Abnormal']):
    path = os.path.join(directory, category)
    save_path = os.path.join(destination, category)
    os.makedirs(save_path, exist_ok=True)
    for subpath in tqdm(os.listdir(path)):
        filePath = os.path.join(path,subpath)
        dicom_files = [os.path.join(filePath, filename) for filename in os.listdir(filePath) if filename.endswith('.dcm')]
        for file_path in dicom_files:
            file_name = os.path.basename(file_path)
            dcm = pydicom.dcmread(file_path)
            try:
                viewposition = dcm[0x0018, 0x5101].value.upper()
                viewposition
                if viewposition != "CC" and viewposition != "MLO":
                    dst_path = os.path.join(save_path, file_name)
                    copyfile(file_path, dst_path)
            except KeyError:
                invalid_count = invalid_count + 1
                dst_path = os.path.join(save_path, file_name)
                if os.path.exists(dst_path):
                    # File with the same name already exists in the destination folder
                    # Rename the file by appending a unique suffix
                    base_name, extension = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(dst_path):
                        new_file_name = f"{base_name}_{counter}{extension}"
                        dst_path = os.path.join(save_path, new_file_name)
                        counter += 1
                copyfile(file_path, dst_path)
print(invalid_count,actual_iamge)