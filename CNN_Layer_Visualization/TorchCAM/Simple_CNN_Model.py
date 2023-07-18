__author__ = "JasonLuo"
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import mlflow
import pandas as pd

LEARNING_RATE = 0.002
mlflow.set_tracking_uri('http://127.0.0.1:1212/?token=5a1ee9f7e18e7103ca55c1abf05876be19605fa431887a32')
class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []
        for label, folder_name in enumerate(['Normal', 'Abnormal']):
            folder_path = os.path.join(root_dir, folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                self.image_list.append(file_path)
                self.label_list.append(label)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.label_list[idx]
        return img, label

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0

def main():
    dataset = PNGDataset(r'C:\Users\Woody\Desktop\Lay_visualization', transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(pretrained=True)

    num_features = model.classifier[6].in_features

    model.classifier[6] = torch.nn.Linear(num_features, 2)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    train_indices, test_indices = train_test_split(list(range(int(len(dataset)))), test_size=0.2, random_state=123)


    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)


    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6)


    num_epochs = 40


    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        model.train()
        for img,label in tqdm(train_dataloader):
            img = img.to(device)

            label = label.to(device)

            outputs = model(img)

            loss = criterion(outputs, label)

            _, preds = torch.max(outputs, 1)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * img.size(0)
            running_corrects += torch.sum(preds == label.data)

        epoch_loss = running_loss / (len(train_dataset)*2)
        epoch_acc = running_corrects.double() / (len(train_dataset)*2)

        mlflow.log_metric('loss', epoch_loss, step=epoch)
        mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()

        y_true = []
        y_pred = []
        # # # evaluate the result at each epoch
        with torch.no_grad():
            for img, label in tqdm(test_dataloader):
                img = img.to(device)

                label = label.to(device)

                outputs = model(img)

                _, preds = torch.max(outputs, 1)

                y_true.extend(label.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                # Calculate evaluation metrics
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            Accuracy = (tp + tn) / (tp + tn + fp + fn)
            print(f"\nF1-Score: {f1:.5f} | Recall: {recall:.2f}")
            print(f"\nSpecificity: {specificity:.5f} | sensitivity: {sensitivity:.5f} | Accuracy: {Accuracy:.2f}%\n")

        mlflow.log_metric('f1', f1, step=epoch)
        mlflow.log_metric('precision', precision, step=epoch)
        mlflow.log_metric('recall', recall, step=epoch)
        mlflow.log_metric('specificity', specificity, step=epoch)
        mlflow.log_metric('sensitivity', sensitivity, step=epoch)
        mlflow.log_metric('test_accuracy', Accuracy, step=epoch)

    scripted_model = torch.jit.script(model)
    mlflow.pytorch.log_model(scripted_model, 'layer-visualiztion', registered_model_name='layer-visualiztion')


if __name__ == '__main__':
    main()
