__author__ = "JasonLuo"
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from Estern_Health.CNNChannels.Module import DualDensennet169
from tqdm.auto import tqdm
import mlflow
from Estern_Health.CNNChannels.Dateset import PNGDataset

# Set the learning rate range for the learning rate finder
LEARNING_RATE = 0.002
mlflow.set_tracking_uri('http://127.0.0.1:5000')
avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0
file = open('/output_final.txt', 'a')
run_names = "run_names"

def main():
    train_dataset = PNGDataset(r'D:\SplitDataSet\train')
    test_dataset = PNGDataset(r'D:\SplitDataSet\test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = DualDensennet169().to(device)
    # model.load_state_dict(torch.load(r'D:/WholeDataTest/Test.pth'))
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6)

    with mlflow.start_run(run_name=run_names):
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

            epoch_loss = running_loss / (len(train_dataset) * 2)
            epoch_acc = running_corrects.double() / (len(train_dataset) * 2)

            mlflow.log_metric('loss', epoch_loss, step=epoch)
            mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            torch.save(model.state_dict(), r'D:/WholeDataTest2'
                                           r'/' + str(epoch) + 'Test.pth')

            model.eval()  # Set the model to evaluation mode
            # #
            y_true = []
            y_pred = []
            predicted_labels = []
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
