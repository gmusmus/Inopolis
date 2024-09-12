import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import logging
from PIL import Image
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Функция для загрузки изображений с обработкой исключений
def pil_loader(path, bad_folder="D://данные//BAD"):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.verify()  # Проверяем, является ли файл изображением
            img = Image.open(f)
            return img.convert('RGB')
    except (IOError, Image.UnidentifiedImageError) as e:
        logging.warning(f"Поврежденный файл изображения {path} будет перемещен. Ошибка: {e}")
        os.makedirs(bad_folder, exist_ok=True)
        shutil.move(path, os.path.join(bad_folder, os.path.basename(path)))
        return None

# Класс для пользовательской загрузки изображений
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if sample is None:
            return torch.zeros(3, 224, 224), target
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Функция создания модели
def create_model(num_classes):
    #model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Функция для аугментации данных
def augmented_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.5),
        transforms.RandomChoice([transforms.RandomRotation(degrees=(-45, -45)),
                                 transforms.RandomRotation(degrees=(45, 45)),
                                 transforms.RandomRotation(degrees=(-30, -30)),
                                 transforms.RandomRotation(degrees=(30, 30))]),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=(20, 20, 20, 20)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.Resize((224, int(224 * 0.8))),
                                transforms.Resize((int(224 * 0.8), 224))], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Функция для загрузки данных
def load_data(data_dir, transform):
    dataset = CustomImageFolder(root=data_dir, transform=transform)
    class_to_idx = dataset.class_to_idx
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, class_to_idx

# Функция для валидации модели
def validate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

# Функция сохранения названий классов
def save_class_names(class_to_idx, save_path):
    df = pd.DataFrame(list(class_to_idx.items()), columns=['Class Name', 'Index'])
    df.to_csv(os.path.join(save_path, 'classes.csv'), index=False)


# Функция для отображения графиков потерь и точности
def plot_training_graphs(train_losses, test_losses, accuracy_list):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def train_model(model, train_loader, test_loader, class_to_idx, epochs, save_dir):
    # Сохранение информации о классах в файл перед началом обучения
    save_class_names(class_to_idx, save_dir)

    # Начало обучения
    train_losses = []
    test_losses = []
    accuracy_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs(save_dir, exist_ok=True)  # проверка что директория для сохранения есть

    start_time = datetime.datetime.now()
    print(f"Начало обучения: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epochs):
        epoch_start_time = datetime.datetime.now()
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, (inputs, labels) in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        test_loss, accuracy = validate_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        accuracy_list.append(accuracy)

        print(
            f'\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        save_path = os.path.join(save_dir,
                                 f'model_STRUCT_DOCUMENT_ResNet50_epocha_{epoch + 1}_acc_{accuracy:.2f}_loss_{test_loss:.4f}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
        }, save_path)
        print(f"\nМодель сохранена: {save_path}")

        epoch_end_time = datetime.datetime.now()
        elapsed_time = epoch_end_time - epoch_start_time
        estimated_remaining_time = (epoch_end_time - start_time) / (epoch + 1) * (epochs - epoch - 1)
        print(f"Время выполнения эпохи: {elapsed_time}, Ориентировочное оставшееся время: {estimated_remaining_time}")

    plot_training_graphs(train_losses, test_losses, accuracy_list)
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"Обучение модели завершено. Общее время обучения: {total_time}")


# Подготовка и запуск обучения модели
data_dir = r"D:\данные"
transform = augmented_transforms()
train_loader, test_loader, class_to_idx = load_data(data_dir, transform)
print(f"Количество классов: {len(class_to_idx)}")
model = create_model(len(class_to_idx))
train_model(model, train_loader, test_loader, class_to_idx,40,r"D:\\данныеМодели")

print("\nРабота скрипта завершена.")


