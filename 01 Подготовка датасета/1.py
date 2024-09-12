
import os  # Импорт модуля os для работы с ОС
import shutil  # Импорт модуля shutil для работы с файлами
from tqdm import tqdm  # Импорт tqdm для отображения прогресса

def get_sd(d):  # Функция для получения подкаталогов
    return [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]  # Возвращаем подкаталоги

def cp_f(src, dst):  # Функция копирования файлов
    f = []  # Список файлов для копирования
    ts = 0  # Общий размер файлов
    for r, _, fs in os.walk(src):  # Проходим по всем файлам
        for f_name in fs:  # Для каждого файла
            fp = os.path.join(r, f_name)  # Полный путь к файлу
            f.append(fp)  # Добавляем в список
            ts += os.path.getsize(fp)  # Увеличиваем общий размер
    tf = len(f)  # Общее число файлов

    pb = tqdm(total=tf, desc=os.path.basename(src), unit="ф", dynamic_ncols=True)  # Прогресс-бар

    for fp in f:  # Для каждого файла в списке
        fn = os.path.basename(fp)  # Получаем имя файла
        t_fp = os.path.join(dst, fn)  # Целевой путь файла
        n = 1  # Начальный номер для дубликатов
        while os.path.exists(t_fp):  # Проверяем наличие файла
            fn_no_ext, ext = os.path.splitext(fn)  # Разделяем имя и расширение
            t_fp = os.path.join(dst, f"{fn_no_ext}({n}){ext}")  # Добавляем номер к имени
            n += 1  # Увеличиваем номер
        try:
            shutil.copy2(fp, t_fp)  # Копируем файл
        except PermissionError as e:  # Ошибка доступа
            print(f"Ошибка доступа {fn}: {e}")  # Вывод сообщения об ошибке
        except Exception as e:  # Любая другая ошибка
            print(f"Ошибка {fn}: {e}")  # Вывод сообщения об ошибке
        pb.update(1)  # Обновляем прогресс-бар
    pb.close()  # Закрываем прогресс-бар

dir_in = r"D:\копия данных"  # Исходный каталог
dir_out = r"D:\данные"  # Целевой каталог

if os.path.exists(dir_out):  # Если целевой каталог существует
    shutil.rmtree(dir_out)  # Удаляем его
os.makedirs(dir_out)  # Создаем заново

for i in get_sd(dir_in):  # Для каждого подкаталога
    os.makedirs(os.path.join(dir_out, i), exist_ok=True)  # Создаем подкаталог в цели
    cp_f(os.path.join(dir_in, i), os.path.join(dir_out, i))  # Копируем файлы
