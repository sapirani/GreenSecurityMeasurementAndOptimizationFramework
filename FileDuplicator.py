import shutil
from pathlib import Path

should_change_content = False
path_of_file_to_duplicate = r"C:\Users\sagib\OneDrive\Desktop\green security\Duplicates\csv files"
Path(path_of_file_to_duplicate).mkdir(parents=True, exist_ok=True)
path_of_directory_to_copy = r"C:\Users\sagib\OneDrive\Desktop\green security\LENOVO Yoga 7 15ITL5 82BJ Windows 10\Balanced Plan\No Scan\Measurement 1\total_memory_each_moment.csv"
DUPLICATE_NUMBER = 1
copied_file_name = "Copy"


def duplicate():
    for i in range(DUPLICATE_NUMBER):
        shutil.copyfile(path_of_file_to_duplicate, f'{copied_file_name}{i}.{path_of_file_to_duplicate.split(".")[-1]}')


def main():
    duplicate()


if __name__ == '__main__':
    main()