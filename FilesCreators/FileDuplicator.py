import os
import shutil
from pathlib import Path
import random
from pathlib import Path
from tqdm import tqdm
import aspose.words as aw
import random
import string


should_change_content = False
file_type = 'pdf'
file_name = 'pdf_file'
path_of_file_to_duplicate = fr"C:\Users\Administrator\Desktop\green security\code\Data\FilesToDuplicate\{file_name}.{file_type}"

DUPLICATE_NUMBER = 10000
MAX_DATA_SIZE = 100
copied_file_name = "Copy"


def change_pdf_content(path_of_origin, path_of_new):
    doc = aw.Document(path_of_origin)
    builder = aw.DocumentBuilder(doc)

    # Insert text at the beginning of the document.
    builder.move_to_document_start()
    text = ''.join(random.choices(string.ascii_lowercase, k=20))
    builder.writeln(text)
    doc.update_page_layout()

    doc.save(path_of_new)


def change_file_content(file_path):
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb+') as f:
        data_size = random.randrange(1, MAX_DATA_SIZE)
        data = os.urandom(data_size)
        offset = random.randrange(1, file_size - data_size)
        f.seek(offset, 0)
        f.write(data)


def duplicate():
    prefix_path_origin = fr"C:\Users\Administrator\Desktop\green security\code\Data\DuplicatedFiles\10000PDForigin"
    prefix_path_changed = fr"C:\Users\Administrator\Desktop\green security\code\Data\DuplicatedFiles\1000PDFchanged"
    Path(prefix_path_origin).mkdir(parents=True, exist_ok=True)
    Path(prefix_path_changed).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(DUPLICATE_NUMBER)):
        new_copy_file_path = f'{prefix_path_origin}\\{copied_file_name}{i}.{file_type}'
        shutil.copy(path_of_file_to_duplicate, new_copy_file_path)
        #new_changed_file_path = f'{prefix_path_changed}\\{copied_file_name}{i}.{file_type}'
        #change_pdf_content(path_of_file_to_duplicate, new_changed_file_path)



def randomly_change():
    prefix_path = fr"C:\Users\Administrator\Desktop\green security\code\Data\DuplicatedFiles\1000PDFchanged"
    Path(prefix_path).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(DUPLICATE_NUMBER)):
        new_file_path = f'{prefix_path}\\{copied_file_name}{i}.{file_type}'
        change_pdf_content(path_of_file_to_duplicate, new_file_path)

        #shutil.copy(path_of_file_to_duplicate, new_file_path)


def main():
    duplicate()


if __name__ == '__main__':
    main()