import os
import shutil
from pathlib import Path
import random
from pathlib import Path
from tqdm import tqdm

# should_change_content = False
file_type = 'pdf'
path_of_file_to_duplicate = fr"data\{file_type}_file.{file_type}"

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
    for should_change_content in range(2):
        if should_change_content:        
            prefix_path = fr"data\{file_type}\changed"
        else:
            prefix_path = fr"data\{file_type}\dup"
        Path(prefix_path).mkdir(parents=True, exist_ok=True)  
        for i in tqdm(range(DUPLICATE_NUMBER)):
            new_file_path = f'{prefix_path}\\{copied_file_name}{i}.{file_type}'
            shutil.copy(path_of_file_to_duplicate, new_file_path)
            if should_change_content:
                change_file_content(new_file_path)


def main():
    duplicate()


if __name__ == '__main__':
    main()