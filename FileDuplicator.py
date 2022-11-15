import os
import shutil
from pathlib import Path
import random

should_change_content = True
path_of_directory_to_copy = r"C:\Users\Administrator\Desktop\University\Green Security Project\GreenSecurity-FirstExperiment\Data\Duplicated Changed Files\power point files"
path_of_file_to_duplicate = r"C:\Users\Administrator\Desktop\University\Green Security Project\GreenSecurity-FirstExperiment\Data\Files To Duplicate\pptx file.pptx"

DUPLICATE_NUMBER = 10000
MAX_DATA_SIZE = 100
copied_file_name = "Copy"


def change_file_content(file_path):
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb+') as f:
        data_size = random.randrange(1, MAX_DATA_SIZE)
        data = os.urandom(data_size)
        offset = random.randrange(1, file_size - data_size)
        f.seek(offset, 0)
        f.write(data)


def duplicate():
    for i in range(DUPLICATE_NUMBER):
        new_file_path = f'{path_of_directory_to_copy}\\{copied_file_name}{i}.{path_of_file_to_duplicate.split(".")[-1]}'
        shutil.copy(path_of_file_to_duplicate, new_file_path)

        if should_change_content:
            change_file_content(new_file_path)


def main():
    duplicate()


if __name__ == '__main__':
    main()