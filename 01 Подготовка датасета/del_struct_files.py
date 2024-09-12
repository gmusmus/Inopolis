import os
import shutil
from tqdm import tqdm

base_dir = r"D:\данные"

# Список подпапок для обработки
folders_to_process = [
    "04 Разрешение на авто"
]

# Список исключений
exceptions = [
    "Модели",
    "01 ПТС",
    "Хлам",
    "00 Не определено",
    "01 ПТС",
    "02 СТС",
    "03 ОСАГО",
    "05 Техосмотр",
    "06 Паспорт",
    "06 Паспорт 1-я страница",
    "06 Паспорт прописка",
    "07 ВУ",
    "08 СНИЛС",
    "09 Судимость",
    "10 ИНН",
    "11 Справки",
]

total_files = 0

# Подсчет общего количества файлов для отображения процента выполнения
for folder in folders_to_process:
    if folder not in exceptions:
        folder_path = os.path.join(base_dir, folder)
        total_files += len(os.listdir(folder_path))

# Проход по каждой папке и перемещение файлов
progress = 0
for folder in folders_to_process:
    if folder not in exceptions:
        folder_path = os.path.join(base_dir, folder)
        files = os.listdir(folder_path)
        for file in tqdm(files, desc=f"Processing files in {folder}", unit="file", leave=False):
            try:
                # Полный путь к файлу
                file_path = os.path.join(folder_path, file)
                # Перемещение файла в базовый каталог
                shutil.move(file_path, os.path.join(base_dir, file))
                progress += 1
                print(f"\rProgress: {progress / total_files * 100:.2f}%", end="")
            except Exception as e:
                print(f"\nError moving file '{file}': {e}")

# Удаление оставшихся пустых папок (если нужно)
for folder in folders_to_process:
    if folder not in exceptions:
        folder_path = os.path.join(base_dir, folder)
        try:
            os.rmdir(folder_path)
        except OSError as e:
            print(f"Error removing folder '{folder}': {e}")

print("\nTask completed.")