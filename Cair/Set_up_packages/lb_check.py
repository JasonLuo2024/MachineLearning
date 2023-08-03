import sys

# Add the path to the specific folder containing Python packages
specific_folder_path = r'/gpfs/home/hluo/anaconda3/envs/myenv/lib'
sys.path.append(specific_folder_path)

import os
import subprocess

required_libraries = [
    'pylibjpeg',
    'torch',
    'concurrent.futures',
    'PIL',
    'threading',
    'pandas',
    'cv2',
    'pydicom',
    'tqdm',
    'multiprocessing',
    'warnings',
    'shutil',
    'skimage',
    'numpy',
    'sklearn',
    'torchvision',
'sklearn.metrics',
'sklearn.model_selection'
]


# Check if each library is installed
output = []
for library in required_libraries:
    try:
        __import__(library)
        output.append(f"{library} is installed.")
    except ImportError:
        output.append(f"{library} is not installed.")

# Save the output to a text file
output_file = r"/gpfs/home/hluo/library_check_output.txt"
with open(output_file, "w") as file:
    file.write("\n".join(output))

print("Library check complete. Output saved to:", output_file)
