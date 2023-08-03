__author__ = "JasonLuo"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import cv2
import StudioGAN
class PNGDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.right_image_list = []
        self.left_image_list = []

        self.left_label_list = []
        self.right_label_list = []

        self.image_list = []
        self.label_list = []

        self.transform = transforms.Compose([
                 transforms.Resize((512, 512)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

        for label, folder_name in enumerate(['Normal', 'Abnormal']):
            folder_path = os.path.join(root_dir, folder_name)

            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)
                self.image_list.append(image_path)
                self.label_list.append(label)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_1 = Image.open(self.image_list[idx]).convert('RGB')

        if self.transform:
            img_1 = self.transform(img_1)

        label_1 = self.label_list[idx]
        return img_1, label_1


def plot_tensor_image(tensor_image):
    # Move the tensor to CPU if it's on GPU
    tensor_image = tensor_image.cpu()

    # Convert the tensor to a NumPy array
    np_image = tensor_image.detach().numpy()

    # Ensure the image has channels in the correct order (C, H, W) for plotting
    np_image = np_image.transpose(1, 2, 0)

    # Display the image
    plt.imshow(np_image)
    plt.axis('off')  # Hide axis
    plt.show()

# Example usage:
# Assuming you have a tensor image named 'fake_images'



# Example usage:
# Assuming you have a tensor image named 'image_tensor'



dataset = PNGDataset(r'C:\Users\Woody\Desktop\Lay_visualization')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# GAN architecture

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, image_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # z = z.view(z.size(0), z.size(1), 1, 1)  # Reshape to 4D tensor
        return self.model(z)





# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model_1 = models.densenet169(pretrained=True)
        self.num_features = self.model_1.classifier.in_features
        self.model_1.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )
    def forward(self, img):
        return self.model_1(img)


torch.manual_seed(42)
np.random.seed(42)

# Initialize generator and discriminator
latent_dim = 100
image_channels = 3
image_size = 512

generator = Generator(latent_dim, image_channels, image_size)
discriminator = Discriminator()

# Loss function and optimizers
generator_loss = torch.nn.BCEWithLogitsLoss()
optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

discriminator_loss = torch.nn.CrossEntropyLoss()
optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=0.002, momentum=0.9)

# Training the GAN

epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to GPU if available
generator.to(device)
discriminator.to(device)
for epoch in range(epochs):
    for img, label in train_loader:
        batch_size = img.size(0)
        real_images = img.to(device)
        real_labels = label.to(device)

        discriminator.zero_grad()
        # Train on real data
        real_output = discriminator(real_images)
        d_loss_real = discriminator_loss(real_output, real_labels)
        d_loss_real.backward(retain_graph=True)
        real_score = real_output.mean().item()


        #Train fake data
        z = torch.randn(batch_size, 3, 512, 512).to(device)
        fake_labels = torch.zeros(batch_size, 1).squeeze(dim=1).to(device)
        fake_labels = fake_labels.long()
        fake_images = generator(z)
        fake_output = discriminator(fake_images)
        d_loss_fake = discriminator_loss(fake_output, fake_labels)
        d_loss_fake.backward(retain_graph=True)
        fake_score = fake_output.mean().item()

        d_loss = d_loss_real + d_loss_fake
        optimizer_discriminator.step()

        # Train Generator

        logits, preds = torch.max(fake_output, dim=1)
        g_loss = generator_loss(logits, real_labels.float())
        generator.zero_grad()
        g_loss.backward(retain_graph=True)
        optimizer_generator.step()

        if epoch % 1 == 0:
            print(
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                  f"D(x): {real_score:.4f} D(G(z)): {fake_score:.4f}")
            # print(fake_images.shape)
            # print(fake_images.squeeze(dim=0).shape)
            # plot_tensor_image(tensor_image=fake_images.squeeze(dim=0))