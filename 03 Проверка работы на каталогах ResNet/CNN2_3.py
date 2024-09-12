# Скрипт для разбирания файлов по классам

import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import csv
import time
from datetime import timedelta
from tqdm import tqdm
import pandas as pd

def read_classes(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            classes = {row['Class Name']: int(row['Index']) for row in reader}
        return classes
    except FileNotFoundError:
        print("Файл не найден.")
        exit()

def load_model(path, num_classes):
    #model = models.resnet18(pretrained=False) # Загрузка архитектуры ResNet-18 без предобученных весов
    model = models.resnet50(pretrained=False)  # Загрузка архитектуры ResNet-50 без предобученных весов
    #model = models.resnet152(pretrained=False) # Загрузка архитектуры ResNet-152 без предобученных весов
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

def transform_image():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def process_images(model, classes, transform, base_dir, model_path, default_threshold=80):
    log_file_path = model_path.replace('.pth', '.log')
    with open(log_file_path, 'w', newline='', encoding='utf-8') as log_file:
        log_writer = csv.writer(log_file)
        header = ['Directory', 'File Name'] + list(classes.keys()) + ['Predicted Class']
        log_writer.writerow(header)

        # Проверка наличия поддиректорий в base_dir
        if any(os.path.isdir(os.path.join(base_dir, d)) for d in os.listdir(base_dir)):
            directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            threshold = default_threshold
        else:
            directories = [base_dir]  # Если поддиректории отсутствуют, обрабатываем корневую директорию
            threshold = 40  # Изменение порога для корневой директории

        for dir_name in directories:
            dir_path = os.path.join(base_dir, dir_name) if dir_name != base_dir else base_dir
            print(f"Обработка: {dir_name}")
            start_time = time.time()
            for img_name in tqdm(os.listdir(dir_path), desc=f"Обработка {dir_name}"):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Проверка на тип файла
                    img_path = os.path.join(dir_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            img_t = transform(img).unsqueeze(0)
                            output = model(img_t)
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
                            max_prob, predicted_idx = torch.max(probabilities, 0)
                            probabilities_list = [f"{prob:.2f}" for prob in probabilities.tolist()]
                            predicted_class_name = list(classes.keys())[list(classes.values()).index(predicted_idx.item())]

                            if max_prob.item() > threshold:
                                target_dir = os.path.join(base_dir, predicted_class_name)
                                os.makedirs(target_dir, exist_ok=True)
                                shutil.move(img_path, os.path.join(target_dir, img_name))

                            log_writer.writerow([dir_path, img_name] + probabilities_list + [predicted_class_name])

                    except Exception as e:
                        print(f"\nОшибка: {img_name}, {e}")

            print(f"\nЗавершение {dir_name}: {timedelta(seconds=time.time() - start_time)}")

def log_to_excel(log_path, excel_path):
    df = pd.read_csv(log_path)
    df.to_excel(excel_path, index=False)


def main():
    base_dir = r"D:\данные\01 ПТС новые"  # Директория с изображениями для обработки
    model_dir = r"D:\данныеМодели\resnet50"  # Директория, где хранится модель и classes.csv

    model_path = os.path.join(model_dir, "model_STRUCT_DOCUMENT_ResNet50_30_acc_95.16_loss_0.1699.pth")
    classes_path = os.path.join(model_dir, "classes.csv")

    classes = read_classes(classes_path)
    model = load_model(model_path, len(classes))
    transform = transform_image()

    process_images(model, classes, transform, base_dir, model_path)

    log_file_path = model_path.replace('.pth', '.log')
    excel_file_path = model_path.replace('.pth', '.xlsx')

    log_to_excel(log_file_path, excel_file_path)
    print(f"Лог-файл сохранен в формате Excel: {excel_file_path}")

if __name__ == "__main__":
    main()
