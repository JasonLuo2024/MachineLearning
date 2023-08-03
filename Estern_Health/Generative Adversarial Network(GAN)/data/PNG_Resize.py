__author__ = "JasonLuo"
from PIL import Image
import os

def resize_images(directory, new_size=(512, 512)):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            img = Image.open(image_path)
            img_resized = img.resize(new_size, Image.ANTIALIAS)
            img_resized.save(os.path.join(directory, filename))

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory containing the images
    directory_path = r"C:\Users\Woody\Desktop\PNG_Data"
    resize_images(directory_path)
