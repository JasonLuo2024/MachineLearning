__author__ = "JasonLuo"
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.right_image_list = []
        self.left_image_list = []

        self.left_label_list = []
        self.right_label_list = []

        self.image_list = []
        self.label_list = []

        for label, folder_name in enumerate(['Normal', 'Abnormal']):
            folder_path = os.path.join(root_dir, folder_name)
            left_path = os.path.join(folder_path, 'CC')
            right_path = os.path.join(folder_path, 'MLO')

            for file_name in os.listdir(left_path):
                image_path = os.path.join(left_path, file_name)
                self.left_image_list.append(image_path)
                self.left_label_list.append(label)

            for file_name in os.listdir(right_path):
                image_path = os.path.join(right_path, file_name)
                self.right_image_list.append(image_path)
                self.right_label_list.append(label)

    def __len__(self):
        return min(len(self.left_image_list),len(self.right_image_list))

    def __getitem__(self, idx):
        img_1 = Image.open(self.left_image_list[idx]).convert('RGB')
        img_2 = Image.open(self.right_image_list[idx]).convert('RGB')

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        label_1 = self.left_label_list[idx]
        label_2 = self.right_label_list[idx]
        return img_1, img_2, label_1, label_2

