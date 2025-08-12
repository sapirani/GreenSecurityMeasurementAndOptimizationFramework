import argparse
import os
import time
import tempfile
from pathlib import Path

from tasks.resources_consumers.task_utils import extract_rate_and_size

TEMP_DIRECTORY_NAME = os.path.join(tempfile.gettempdir(), "disk_io_test")
BASE_FILE_NAME = "file"
FILE_ENDING = "txt"
FILE_SIZE = 1024
NUM_FILES = 100


def prepare_test_files(directory: str, file_size: int = FILE_SIZE, num_files: int = NUM_FILES):
    """Create a few files with given size so we can read from them."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    data = os.urandom(file_size)
    for i in range(num_files):
        with open(os.path.join(directory, f"{BASE_FILE_NAME}{i}.{FILE_ENDING}"), 'wb') as f:
            f.write(data)


def read_files(rate: float, file_size: int):
    """Read files endlessly at a given rate and read size (bytes)."""
    # Create temp dir for test files
    if not os.path.exists(TEMP_DIRECTORY_NAME) or not os.listdir(TEMP_DIRECTORY_NAME):
        prepare_test_files(TEMP_DIRECTORY_NAME, file_size)

    all_files = [os.path.join(TEMP_DIRECTORY_NAME, f) for f in os.listdir(TEMP_DIRECTORY_NAME)
                 if os.path.isfile(os.path.join(TEMP_DIRECTORY_NAME, f))]

    interval = 1.0 / rate
    file_index = 0

    print(f"Reading {file_size} bytes per file at {rate} reads/sec from {TEMP_DIRECTORY_NAME}")
    try:
        while True:
            start_time = time.time()
            file_path = all_files[file_index]
            try:
                with open(file_path, 'rb') as f:
                    f.read(file_size)
            except Exception:
                pass

            file_index = (file_index + 1) % len(all_files)

            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time >= 0:
                time.sleep(sleep_time)
            else:
                raise RuntimeError("Received a negative sleep time. The Rate value is too high.")
    except KeyboardInterrupt:
        print("\nStopping disk read operations...")


if __name__ == '__main__':
    task_description = "Performs Disk I/O read operations endlessly at a given rate and read size."
    rate, file_size = extract_rate_and_size(task_description, FILE_SIZE)

    read_files(
        rate=rate,
        file_size=file_size
    )
