import os

def clean_empty_subfolders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if not os.listdir(folder_path):
                print(f"Removing empty subfolder: {folder_path}")
                os.rmdir(folder_path)

# Example usage
directory_path = r"D:\WholeDataSet\Normal"
clean_empty_subfolders(directory_path)