import os
import fitz  # PyMuPDF
from tqdm import tqdm

def convert_pdf(d, o, dpi=300):
    # Подсчитываем общее число страниц N
    N = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".pdf"):
                with fitz.open(os.path.join(r, f)) as doc:
                    N += len(doc)

    # Масштабируем, чтобы достичь желаемого DPI
    s = dpi / 72  # Для 300 DPI это ~4.17

    # Инициализируем прогресс-бар для N страниц
    p = tqdm(total=N, desc="Прогресс конвертации", unit="стр")

    # Итерируем по всем файлам
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".pdf"):
                pth = os.path.join(r, f)
                doc = fitz.open(pth)

                # Обрабатываем каждую страницу
                for i in range(len(doc)):
                    pg = doc.load_page(i)
                    m = fitz.Matrix(s, s)  # Матрица масштабирования
                    px = pg.get_pixmap(matrix=m)

                    # Сохраняем с масштабированием в o
                    out_f = os.path.join(o, f"{os.path.splitext(f)[0]}_p{i+1}.jpg")
                    px.save(out_f)

                    p.update(1)  # Обновляем прогресс

                doc.close()

    p.close()  # Завершаем отслеживание прогресса


# Использование
d = r"D:\данные для распознования"  # Исходная директория
o = r"D:\куда_сохраняем"  # Директория для сохранения
convert_pdf(d, o)
print("Конец")
