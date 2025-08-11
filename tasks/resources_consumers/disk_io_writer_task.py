import argparse
import os
from pathlib import Path
import random
import string

from utils.general_consts import KB

NUMBER_OF_FILES = 10000
BASE_FILE_NAME = "file"
FILE_ENDING = "txt"

RANDOM_STRING_LEN = 0.7 * KB
RANDOM_STRING = ''.join([random.choice(string.ascii_letters) for i in range(RANDOM_STRING_LEN)])


def write_files(files_directory: str, number_of_files: int = NUMBER_OF_FILES):
    Path(files_directory).mkdir(parents=True, exist_ok=True)
    for i in range(number_of_files):
        file_path = os.path.join(files_directory, f"{BASE_FILE_NAME}{i}.{FILE_ENDING}")
        generate_file(file_path)


def generate_file(file_path: str):
    with open(f'{file_path}', 'w') as f:
        f.write(RANDOM_STRING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only writes and consumes Disk"
    )

    parser.add_argument("-n", "--number_of_files",
                        type=int,
                        default=NUMBER_OF_FILES,
                        help="The number of files to generate.")

    parser.add_argument("-d", "--directory",
                        type=str,
                        required=True,
                        help="The path to the directory where the generated files will be saved.")

    args = parser.parse_args()
    write_files(args.directory, args.number_of_files)
