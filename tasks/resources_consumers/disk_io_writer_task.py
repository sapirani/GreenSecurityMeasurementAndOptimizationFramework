import os
import time
import random
import string
from pathlib import Path
import tempfile

from tasks.resources_consumers.task_utils import extract_rate_and_size

TEMP_OUTPUT_DIRECTORY = os.path.join(tempfile.gettempdir(), "disk_io_test")
BASE_FILE_NAME = "file"
FILE_ENDING = "txt"
FILE_SIZE = 1024


def generate_random_string(size: int) -> str:
    """Generate a random string of given byte size."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(size))


def write_files(rate: float, file_size: int = FILE_SIZE):
    """Write files endlessly at a given rate (files/sec) and file size."""
    # Create a temporary directory for writing
    Path(TEMP_OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    interval = 1.0 / rate
    counter = 0
    data = generate_random_string(file_size)

    print(f"Writing {file_size} bytes per file at {rate} files/sec in {TEMP_OUTPUT_DIRECTORY}")
    try:
        while True:
            start_time = time.time()
            file_path = os.path.join(TEMP_OUTPUT_DIRECTORY, f"{BASE_FILE_NAME}{counter}.{FILE_ENDING}")
            with open(file_path, 'w') as f:
                f.write(data)
            counter += 1

            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time >= 0:
                time.sleep(sleep_time)
            else:
                raise RuntimeError("Received a negative sleep time. The Rate value is too high.")
    except KeyboardInterrupt:
        print("\nStopping disk write operations...")


if __name__ == '__main__':
    task_description = "Performs Disk I/O write operations endlessly at a given rate and file size."
    rate, file_size = extract_rate_and_size(task_description, FILE_SIZE)

    write_files(
        rate=rate,
        file_size=file_size
    )
