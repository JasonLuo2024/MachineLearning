import os
import random
from shutil import copyfile
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

def split_dataset_folder(dataset_folder,SplitDataSet_folder_train,SplitDataSet_folder_test, test_size=0.2, random_seed=42):
    # Create the train and test folders if they don't exist
    for category in tqdm(['Normal', 'Abnormal']):
        dataset_path = os.path.join(dataset_folder, category)
        train_path = os.path.join(SplitDataSet_folder_train, category)
        test_path = os.path.join(SplitDataSet_folder_test, category)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        for view in ["MLO","CC"]:
            file_path = os.path.join(dataset_path,view)
            train_save_path = os.path.join(train_path, view)
            test_save_path = os.path.join(test_path, view)
            os.makedirs(train_save_path, exist_ok=True)
            os.makedirs(test_save_path, exist_ok=True)
            file_names = os.listdir(file_path)
            train_file_names, test_file_names = train_test_split(file_names, test_size=test_size,
                                                                 random_state=random_seed)
            for file_name in train_file_names:
                src_path = os.path.join(file_path, file_name)
                dst_path = os.path.join(train_save_path, file_name)
                copyfile(src_path, dst_path)

            for file_name in test_file_names:
                src_path = os.path.join(file_path, file_name)
                dst_path = os.path.join(test_save_path, file_name)
                copyfile(src_path, dst_path)


# Usage example
dataset_folder = 'D:/WholeDateTest'  # Specify the path to your dataset folder
SplitDataSet_folder_test = 'D:/SplitDataSet/test'      # Specify the path where the train dataset will be created
SplitDataSet_folder_train = 'D:/SplitDataSet/train'
split_dataset_folder(dataset_folder, SplitDataSet_folder_train,SplitDataSet_folder_test, test_size=0.2, random_seed=42)
