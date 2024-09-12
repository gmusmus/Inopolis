import os
import fitz  # PyMuPDF
from tqdm import tqdm


def convert_pdf_pages_to_jpg_with_dpi(pdf_dir, dpi=300):
    # Коэффициент масштабирования для достижения 300 DPI
    scale = dpi / 72  # Примерно 4.17 для 300 DPI

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc = fitz.open(pdf_path)

            pbar = tqdm(total=len(doc), desc=f"Конвертация {filename}", unit="стр", leave=False)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Создаем матрицу преобразования с нужным масштабированием для 300 DPI
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)

                output_file = os.path.join(pdf_dir, f"{os.path.splitext(filename)[0]}_page_{page_num + 1}.jpg")
                pix.save(output_file)

                pbar.update(1)

            pbar.close()
            doc.close()


# Пример использования
pdf_dir = "D:\\данные\\06 Паспорт"
convert_pdf_pages_to_jpg_with_dpi(pdf_dir)