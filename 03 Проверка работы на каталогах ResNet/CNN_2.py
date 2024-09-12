import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm


def load_model(model_path, num_classes=2):
    #model = models.resnet18(pretrained=False) # Загрузка архитектуры ResNet-18 без предобученных весов
    model = models.resnet50(pretrained=False)  # Загрузка архитектуры ResNet-50 без предобученных весов
    #model = models.resnet152(pretrained=False) # Загрузка архитектуры ResNet-152 без предобученных весов
    num_ftrs = model.fc.in_features  # Получаем количество входных признаков последнего слоя
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Заменяем последний слой на новый с num_classes выходами
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])  # Загружаем сохраненные веса модели
    model.eval()  # Переводим модель в режим оценки
    return model  # Возвращаем загруженную модель

def process_images(model, data_dir, target_dir, transform):
    # Проверяем наличие файлов в исходном каталоге
    if not os.listdir(data_dir):
        print(f"Каталог {data_dir} пустой. Пропускаем...")  # Сообщаем, если каталог пуст
        return

    if not os.path.exists(target_dir):  # Проверяем существование целевого каталога
        os.makedirs(target_dir)  # Создаем целевой каталог, если он не существует

    classes = ['class_0', 'class_1']  # Определяем названия классов
    for class_name in classes:  # Создаем подкаталоги для каждого класса
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

    # Цикл обработки изображений
    for img_name in tqdm(os.listdir(data_dir), desc="Обработка фотографий"):  # Итерируемся по файлам в исходном каталоге
        img_path = os.path.join(data_dir, img_name)
        try:
            with Image.open(img_path) as img:  # Открываем изображение
                img = transform(img).unsqueeze(0)  # Применяем преобразование и добавляем размерность пакета
                output = model(img)  # Получаем вывод модели
                _, predicted = torch.max(output, 1)  # Определяем предсказанный класс
                class_name = classes[predicted.item()]  # Получаем название класса по индексу
                shutil.move(img_path, os.path.join(target_dir, class_name, img_name))  # Перемещаем файл в соответствующий подкаталог
        except Exception as e:
            print(f"Ошибка обработки файла {img_name}: {e}")  # Выводим сообщение об ошибке, если она произошла



def main():
    model_path = r"D:\данные\Модели\model_STRUCT_DOCUMENT_ResNet50_epocha_40_acc_97.80_loss_0.0738.pth"  # Путь к файлу модели

    base_dir = r"D:\данные"  # Базовый каталог для обработки изображений

    # Список подпапок для обработки. Каждая подпапка содержит определенный тип документов
    # folders_to_process=["04 Разрешение на авто"]
    folders_to_process = [
        "00 Не определено",
        "01 ПТС 1страница",
        "01 ПТС 2страница",
        "02 СТС 1страница",
        "02 СТС 2страница",
        "03 ОСАГО 1страница",
        "03 ОСАГО 2страница",
        "04 Разрешение на авто новое",
        "04 Разрешение старое",
        "05 Техосмотр",
        "06 Паспорт",
        "06 Паспорт 1-я страница",
        "06 Паспорт прописка",
        "07 ВУ 1 страница",
        "07 ВУ 2 страница",
        "07 ВУ старые права",
        "08 СНИЛС 1 страница",
        "08 СНИЛС 2 страница",
        "09 Судимость",
        "10 ИНН",
        "Модели",
        "Хлам"
    ]

    # Список каталогов, которые будут исключены из обработки
    exceptions = [
        "00 Не определено",
        "Модели",
        "Хлам"
    ]


    # Определяем трансформации для предобработки изображений перед подачей в модель
    transform = transforms.Compose([
        transforms.Resize(256),  # Изменение размера изображения
        transforms.CenterCrop(224),  # Обрезка изображения до центра
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])

    model = load_model(model_path)  # Загрузка модели

    for folder in folders_to_process:  # Перебор папок для обработки
        if folder not in exceptions:  # Пропуск исключенных папок
            data_dir = os.path.join(base_dir, folder)  # Формирование пути к папке с документами

            # Обработка изображений в папке, классификация и сортировка по категориям
            process_images(model, data_dir, data_dir, transform)



if __name__ == "__main__":
    main()