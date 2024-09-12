import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import csv
import time
from datetime import timedelta
from tqdm import tqdm

# Функция для чтения количества классов
def read_num_classes(classes_csv_path):
    try:
        with open(classes_csv_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            num_classes = sum(1 for row in reader) - 1  # Вычитаем 1 для заголовка
            return num_classes
    except FileNotFoundError:
        print("Файл classes.csv не найден.")
        exit()

def load_model(model_path, num_classes):
    #model = models.resnet18(pretrained=False) # Загрузка архитектуры ResNet-18 без предобученных весов
    model = models.resnet50(pretrained=False)  # Загрузка архитектуры ResNet-50 без предобученных весов
    #model = models.resnet152(pretrained=False) # Загрузка архитектуры ResNet-152 без предобученных весов
    num_ftrs = model.fc.in_features  # Получаем количество входных признаков последнего слоя
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Заменяем последний слой на новый с num_classes выходами
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])  # Загружаем сохраненные веса модели
    model.eval()  # Переводим модель в режим оценки
    return model  # Возвращаем загруженную модель

# Функция чтения классов и создания словаря
def read_classes(classes_csv_path):
    classes = {}
    index_to_class = {}
    try:
        with open(classes_csv_path, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                classes[row['Class Name']] = int(row['Index'])
                index_to_class[int(row['Index'])] = row['Class Name']
        return classes, index_to_class
    except FileNotFoundError:
        print("Файл classes.csv не найден.")
        exit()

# Функция определения трансформаций изображений
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Функция записи шапки файла лога
def create_log_file_header(log_file_path, classes):
    header = ['File Path', 'File Name', 'Predicted Class Name', 'Best Class Index']
    for class_name in classes.keys():
        header.append(f"Prob {class_name} (%)")
    header.append('Best Class Prob (%)')

    with open(log_file_path, mode='w', newline='', encoding='utf-8') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(header)

# Функция обработки одного изображения
def process_image(model, img_path, transform, classes, index_to_class, base_dir, confidence_threshold=80):
    try:
        with Image.open(img_path) as img:
            img_t = transform(img).unsqueeze(0)
            output = model(img_t)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            max_prob, predicted = torch.max(probabilities, 0)
            predicted_class_name = index_to_class[predicted.item()]
            predicted_class_index = predicted.item()
            probs_formatted = [f"{prob:.2f}" for prob in probabilities.tolist()]
            result = [img_path, os.path.basename(img_path), predicted_class_name,
                      predicted_class_index] + probs_formatted + [f"{max_prob.item():.2f}"]

            # Путь к поддиректории для класса с наивысшей вероятностью
            target_dir = os.path.join(base_dir, predicted_class_name)
            os.makedirs(target_dir, exist_ok=True)
            # Перемещение файла
            shutil.move(img_path, os.path.join(target_dir, os.path.basename(img_path)))
            result.append('Moved')

            return result
    except Exception as e:
        print(f"Ошибка обработки файла {os.path.basename(img_path)}: {e}")
        return None




def process_images_in_directory(model, base_dir, class_name, transform, classes, index_to_class, log_file_path, confidence_threshold):
    data_dir = os.path.join(base_dir, class_name)

    # Создание каталога, если он не существует
    if not os.path.exists(data_dir):
        print(f"Каталог {class_name} не существует. Создаем...")
        os.makedirs(data_dir)
    # Проверка на пустоту каталога
    elif not os.listdir(data_dir):
        print(f"Каталог {class_name} пуст. Пропускаем...")
        return

    start_time = time.time()
    print(f"Начало обработки {class_name}: {time.ctime(start_time)}")

    with open(log_file_path, mode='a', newline='', encoding='utf-8') as log_file:
        log_writer = csv.writer(log_file)
        for img_name in tqdm(os.listdir(data_dir), desc=f"Обработка {class_name}"):
            img_path = os.path.join(data_dir, img_name)
            result = process_image(model, img_path, transform, classes, index_to_class, confidence_threshold)
            if result:
                log_writer.writerow(result[:-1])  # Не записывать статус перемещения в лог

    end_time = time.time()
    print(f"Обработка {class_name} завершена за {timedelta(seconds=end_time - start_time)}")

# Основной цикл обработки
def main(model_path, base_dir, classes_csv_path):
    num_classes = read_num_classes(classes_csv_path)
    model = load_model(model_path, num_classes)
    classes, index_to_class = read_classes(classes_csv_path)
    print(classes,index_to_class)
    transform = get_transforms()

    log_file_path = os.path.splitext(model_path)[0] + '_log.csv'
    create_log_file_header(log_file_path, classes)

    for class_name in classes:
        process_images_in_directory(model, base_dir, class_name, transform, classes, index_to_class, log_file_path,80)

if __name__ == "__main__":
    model_path = r"D:\данныеМодели\ResNet50-2\model_STRUCT_DOCUMENT_ResNet50_epocha_40_acc_97.80_loss_0.0738.pth"
    base_dir = r"D:\данные\07 ВУ 1 страница"
    classes_csv = r"D:\данныеМодели\ResNet50-2\classes.csv"
    main(model_path, base_dir, classes_csv)