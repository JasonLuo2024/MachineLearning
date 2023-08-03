__author__ = "JasonLuo"
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import mlflow
from Dateset import PNGDataset

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
            torch.nn.Linear(512, 2)
        )

        self.model_2.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x1, x2):
        output1 = self.model_1(x1)

        output2 = self.model_2(x2)

        output = torch.cat((output1, output2), dim=0)

        return output


# Set the learning rate range for the learning rate finder
LEARNING_RATE = 0.002
mlflow.set_tracking_uri('http://127.0.0.1:5000')
avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0
file = open('output_final.txt', 'a')

def main():
    train_dataset = PNGDataset(r'D:\SplitDataSet\train')
    test_dataset = PNGDataset(r'D:\SplitDataSet\test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



        # Modify the classifier

    model = DualDensennet169().to(device)

    # model.load_state_dict(torch.load(r'D:/WholeDataTest/Test.pth'))


    # Update the model state dictionary key names if needed
    # (e.g., remove 'module.' prefix if using DataParallel)


    criterion = torch.nn.CrossEntropyLoss()



    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # Split the data into train and test sets

    # train_indices, test_indices = train_test_split(list(range(int(len(dataset)))), test_size=0.2, random_state=123)



    # train_labels = []
    # for idx in range(len(train_indices)):
    #     train_labels.append(dataset.left_label_list[train_indices[idx]])
    #     train_labels.append(dataset.right_label_list[train_indices[idx]])
    #
    #
    # test_labels = []
    # for idx in range(len(test_indices)):
    #     test_labels.append(dataset.left_label_list[test_indices[idx]])
    #     test_labels.append(dataset.right_label_list[test_indices[idx]])
    #
    #
    # image_path = []
    # for idx in range(len(test_indices)):
    #     image_path.append(dataset.left_image_list[test_indices[idx]])
    #     image_path.append(dataset.right_image_list[test_indices[idx]])
    #
    #
    # train_labels_count = [train_labels.count(label) for label in range(2)]
    # test_labels_count = [test_labels.count(label) for label in range(2)]

    # CC&MLO list using the same index, then the length would be len(CC) + len(MLO)
    # print("Train set samples:", len(train_indices)*2)
    # print("Train set labels count:", train_labels_count)
    #
    # print("Test set samples:", len(test_indices)*2)
    # print("Test set labels count:", test_labels_count)
    #
    # print("Train set samples:", len(train_indices)*2)
    # print("Test set samples:", len(test_indices)*2)
    #
    # train_dataset = torch.utils.data.Subset(dataset, train_indices)
    # test_dataset = torch.utils.data.Subset(dataset, test_indices)


    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6)



    num_epochs = 40


    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for img_1, img_2, label_1, label_2 in tqdm(train_dataloader):
            model.train()
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            label_1 = label_1.to(device)
            label_2 = label_2.to(device)
            label = torch.cat((label_1, label_2), dim=0)


            outputs = model(img_1, img_2)


            loss = criterion(outputs, label)

            _, preds = torch.max(outputs, 1)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            running_loss += loss.item() * (img_1.size(0) + img_2.size(0))
            running_corrects += torch.sum(preds == label.data)

        epoch_loss = running_loss / (len(train_dataset)*2)
        epoch_acc = running_corrects.double() / (len(train_dataset)*2)

        mlflow.log_metric('loss', epoch_loss, step=epoch)
        mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        torch.save(model.state_dict(), r'D:/WholeDataTest2'
                                       r'/'+str(epoch)+'Test.pth')

        model.eval()  # Set the model to evaluation mode
        # #
        y_true = []
        y_pred = []
        predicted_labels = []
        # # # evaluate the result at each epoch
        with torch.no_grad():
            for img_1, img_2, label_1, label_2 in tqdm(test_dataloader):
                img_1 = img_1.to(device)
                img_2 = img_2.to(device)
                label_1 = label_1.to(device)
                label_2 = label_2.to(device)
                labels = torch.cat((label_1, label_2), dim=0)

                outputs = model(img_1, img_2)

                _, preds = torch.max(outputs, 1)
                for pred in preds.tolist():
                    predicted_labels.append(pred)

                y_true.extend(labels.cpu().numpy())
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
    mlflow.pytorch.log_model(scripted_model, 'breast-cancer-model', registered_model_name='breast-cancer-model')


if __name__ == '__main__':
    main()
