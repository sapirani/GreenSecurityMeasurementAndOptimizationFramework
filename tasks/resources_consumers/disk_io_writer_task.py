import argparse
from pathlib import Path
import random
import string

from utils.general_consts import KB

NUMBER_OF_FILES = 10000
BASE_FILE_NAME = "file"
FILE_ENDING = "txt"


def rand_letters(size: int) -> str:
    return ''.join([random.choice(string.ascii_letters) for i in range(size)])


def write_files(files_directory: str, number_of_files: int = NUMBER_OF_FILES):
    Path(files_directory).mkdir(parents=True, exist_ok=True)
    for i in range(number_of_files):
        generate_file(f'{files_directory}\\{BASE_FILE_NAME}{i}.{FILE_ENDING}')


def generate_file(file_path: str):
    with open(f'{file_path}', 'w') as f:
        f.write(rand_letters(random.randint(0.5 * KB, KB)))

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