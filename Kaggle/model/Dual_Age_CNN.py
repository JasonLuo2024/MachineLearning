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
import pandas as pd
import mlflow

# Set the learning rate range for the learning rate finder
LEARNING_RATE = 0.002
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.CC_image_list = []
        self.MLO_image_list = []

        self.CC_label_list = []
        self.MLO_label_list = []

        self.CC_age_list = []
        self.MLO_age_list = []

        for label, folder_name in enumerate(['Normal', 'Abnormal']):
            folder_path = os.path.join(root_dir, folder_name)
            CC_path = os.path.join(folder_path, 'CC')
            MLO_path = os.path.join(folder_path, 'MLO')

            for folder_path in [CC_path, MLO_path]:
                age_list = self.CC_age_list if folder_path == CC_path else self.MLO_age_list
                label_list = self.CC_label_list if folder_path == CC_path else self.MLO_label_list
                image_list = self.CC_image_list if folder_path == CC_path else self.MLO_image_list
                for root, directories, files in os.walk(folder_path):
                    for file in files:
                        folder_name = os.path.basename(root)
                        number = int(folder_name.split("\\")[-1])
                        age_list.append(number)

                        file_path = os.path.join(root, file)

                        image_list.append(file_path)

                        label_list.append(label)


    def __len__(self):
        return min(len(self.CC_image_list),len(self.MLO_image_list))

    def __getitem__(self, idx):
        img_1 = Image.open(self.CC_image_list[idx]).convert('RGB')
        img_2 = Image.open(self.MLO_image_list[idx]).convert('RGB')

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        label_1 = self.CC_label_list[idx]
        label_2 = self.MLO_label_list[idx]
        age_1 = self.CC_age_list[idx]
        age_2 = self.MLO_age_list[idx]
        return img_1, img_2, label_1, label_2, age_1, age_2, self.CC_image_list[idx],self.MLO_image_list[idx]


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0
file = open('output_final.txt', 'a')
# Define a custom print function
def custom_print(*args, **kwargs):
    # Convert all arguments to strings
    strings = [str(arg) for arg in args]
    # Join the strings with a space separator
    output = ' '.join(strings)
    # Write the output to the file
    file.write(output + '\n')
    # Print the output to the console
    print(output, **kwargs)

# Replace the default print function with the custom print function


def main():
    dataset = PNGDataset(r'C:\Users\Woody\Desktop\RSNA_PNG', transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    class DualDensennet169(nn.Module):
        def __init__(self):
            super(DualDensennet169, self).__init__()
            self.model_1 = models.densenet169(pretrained=True)
            self.model_2 = models.densenet169(pretrained=True)
            self.num_features = self.model_1.classifier.in_features


            self.model_1.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.num_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
            )

            self.model_2.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.num_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
            )

            self.age_linear = torch.nn.Linear(1, 64)
            self.relu  = torch.nn.ReLU()
            self.fc = torch.nn.Linear(576,2)
        def forward(self, x1, x2,age1,age2):
            age_embedding1 = self.age_linear(age1.unsqueeze(-1).to(self.age_linear.weight.dtype))
            age_embedding2 = self.age_linear(age2.unsqueeze(-1).to(self.age_linear.weight.dtype))

            # Get the outputs from the image branches
            output1 = self.relu(self.model_1(x1))
            output2 = self.relu(self.model_2(x2))

            # Concatenate the age embeddings with the outputs
            output1 = torch.cat((output1, age_embedding1), dim=1)
            output2 = torch.cat((output2, age_embedding2), dim=1)
            output1 = self.fc(output1)
            output2 = self.fc(output2)

            # Concatenate the outputs from both branches
            output = torch.cat((output1, output2), dim=0)

            return output

        # Modify the classifier

    model = DualDensennet169().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    train_indices, test_indices = train_test_split(list(range(int(len(dataset)))), test_size=0.2, random_state=123)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False,num_workers=8)

    num_epochs = 40

    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        try:
            for img_1, img_2, label_1, label_2, age_1, age_2,path_1,path_2 in tqdm(train_dataloader):
                model.train()
                img_1 = img_1.to(device)
                img_2 = img_2.to(device)
                label_1 = label_1.to(device)
                label_2 = label_2.to(device)
                age_1 = age_1.to(device)
                age_2 = age_2.to(device)

                label = torch.cat((label_1, label_2), dim=0)

                outputs = model(img_1, img_2, age_1, age_2)

                loss = criterion(outputs, label)

                _, preds = torch.max(outputs, 1)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                running_loss += loss.item() * (img_1.size(0) + img_2.size(0))
                running_corrects += torch.sum(preds == label.data)

            epoch_loss = running_loss / (len(train_dataset) * 2)
            epoch_acc = running_corrects.double() / (len(train_dataset) * 2)

            mlflow.log_metric('loss', epoch_loss, step=epoch)
            mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # torch.save(model.state_dict(), r'D:/RSNA_PNG/' + str(epoch) + 'Test.pth')

            model.eval()  # Set the model to evaluation mode
            # #
            y_true = []
            y_pred = []

            test_labels = []
            predicted_labels = []
            image_path = []
            patient_ID = []

            train_csv = r'C:\Users\Woody\Desktop\RSNA_PNG\train.csv'
            # # # evaluate the result at each epoch

            with torch.no_grad():
                for img_1, img_2, label_1, label_2, age_1, age_2,path_1,path_2 in tqdm(test_dataloader):
                    for item in [label_1,label_2]:
                        for label in item:
                          test_labels.append(label)

                    img_1 = img_1.to(device)
                    img_2 = img_2.to(device)
                    label_1 = label_1.to(device)
                    label_2 = label_2.to(device)
                    age_1 = age_1.to(device)
                    age_2 = age_2.to(device)

                    labels = torch.cat((label_1, label_2), dim=0)
                    outputs = model(img_1, img_2, age_1, age_2)

                    _, preds = torch.max(outputs, 1)
                    #

                    for item in [path_1,path_2]:
                        for path in item:
                            image_id = os.path.splitext(os.path.basename(path))[0].split('.')[0]
                            image_path.append(os.path.dirname(path)+image_id)
                            df = pd.read_csv(train_csv, delimiter=',')
                            # Filter the DataFrame based on the patient's ID
                            filtered_df = df[df['image_id'] == int(image_id)]
                            if not filtered_df.empty:
                                id = str(int(filtered_df.iloc[0]['patient_id']))
                                patient_ID.append(id)


                    for pred in preds.tolist():
                        predicted_labels.append(pred)


                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            df = pd.DataFrame({
                        'Test Labels': test_labels,
                        'Predicted Labels': predicted_labels,
                        'Patient ID': patient_ID,
                        'Image Path': image_path
            })
            df.to_csv('output_' + str(epoch) + '.csv', index=False)

            # Calculate evaluation metrics
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            Accuracy = (tp + tn) / (tp + tn + fp + fn)


            output_file = '/home/hluo/results'+str(epoch)+ '/metrics.txt'
            with open(output_file, "w") as file:
                file.write(f"F1-Score: {f1:.5f} | Recall: {recall:.2f}\n")
                file.write(
                    f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}%\n")

            print(f"\nF1-Score: {f1:.5f} | Recall: {recall:.2f}")
            print(f"\nSpecificity: {specificity:.5f} | sensitivity: {sensitivity:.5f} | Accuracy: {Accuracy:.2f}%\n")

            mlflow.log_metric('f1', f1, step=epoch)
            mlflow.log_metric('precision', precision, step=epoch)
            mlflow.log_metric('recall', recall, step=epoch)
            mlflow.log_metric('specificity', specificity, step=epoch)
            mlflow.log_metric('sensitivity', sensitivity, step=epoch)
            mlflow.log_metric('test_accuracy', Accuracy, step=epoch)
        except Exception as e:
            print(e)


    scripted_model = torch.jit.script(model)
    mlflow.pytorch.log_model(scripted_model, 'breast-cancer-model', registered_model_name='breast-cancer-model_age')


if __name__ == '__main__':
    main()