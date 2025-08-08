import os
from pathlib import Path


def read_files(directory_path: str):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    all_files_and_directories = os.listdir(directory_path)
    all_files = [f for f in all_files_and_directories if
                 os.path.isfile(os.path.join(directory_path, f))]  # Filtering only the files.
    for file in all_files:
        file_path = os.path.join(directory_path, file)
        try:
            with open(file_path, 'rb') as f:
                # print(f.name)
                x = f.read()
        except Exception as e:
            continue
