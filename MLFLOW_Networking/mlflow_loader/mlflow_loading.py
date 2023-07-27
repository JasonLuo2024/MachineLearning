__author__ = "JasonLuo"
import os
import tqdm
import re
import mlflow
def file_name_change(folder_path = ''):
    folder_path = folder_path
    Sort_list = []
    for root, directories, files in os.walk(folder_path):
        for file in files:
            Sort_list.append(file)
            num = file.split('.')[0] + '.txt'
            new_filename = os.path.join(root, num)
            old_filename = os.path.join(root, file)
            os.rename(old_filename, new_filename)
def extract_file_number(file_path):
    file_name = os.path.basename(file_path)
    match = file_name.split('.')[0]
    return int(match)

def read_metrics_from_file(file_path):
    metrics = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(': ')
            if len(parts) == 2:
                metric_name = parts[0].strip()
                metric_value = parts[1].split(' |')[0].strip()
                metrics[metric_name] = float(metric_value)

    return metrics




LEARNING_RATE = 0.002
mlflow.set_tracking_uri('http://127.0.0.1:5000')
folder_path = r'C:\Users\Woody\Desktop\result\Batch_16_without_age\result'
file_name_change(folder_path)
file_path = []
for root, directories, files in os.walk(folder_path):
    for file in files:
        filename = os.path.join(root, file)
        file_path.append(filename)

sorted_file_paths = sorted(file_path, key=extract_file_number)
# print(sorted_file_paths)
for file_path in sorted_file_paths:
    with open(file_path, 'r') as file:
        epoch = int(os.path.basename(file_path).split('.')[0])
        value_list = []
        for line in file:
            item_list = line.split('|')
            for item in item_list:
                value_list.append(item)

        mlflow.log_metric('loss', float(value_list[0].split(':')[1]), step=epoch)
        mlflow.log_metric('train_accuracy', float(value_list[1].split(':')[1]), step=epoch)
        mlflow.log_metric('f1', float(value_list[2].split(':')[1]), step=epoch)
        mlflow.log_metric('recall', float(value_list[3].split(':')[1]), step=epoch)
        mlflow.log_metric('specificity', float(value_list[4].split(':')[1]), step=epoch)
        mlflow.log_metric('sensitivity', float(value_list[5].split(':')[1]), step=epoch)

print('load successfully')






    # metrics = read_metrics_from_file(file_path)
    #
    # if metrics:
    #     for metric_name, metric_value in metrics.items():
    #         print(f"{metric_name}: {metric_value:.5f}")
    # else:
    #     print("No metrics found in the file.")
