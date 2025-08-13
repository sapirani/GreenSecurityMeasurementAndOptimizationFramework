import time

from tasks.resources_consumers.task_utils import extract_rate_and_size
from utils.general_consts import GB

DEFAULT_CHUNK_SIZE_IN_BYTES = 2 ** 10  # 1 KB
INITIAL_BUFFER_SIZE = 5 * GB
BUFFER = bytearray(INITIAL_BUFFER_SIZE)


def release_ram(rate: float, chunk_size: int = DEFAULT_CHUNK_SIZE_IN_BYTES):
    """
    Gradually releases memory from a pre-allocated buffer
    at the given rate (chunks/sec) and chunk size.
    """
    done_release = False
    while not done_release:
        start_time = time.time()

        for _ in range(int(rate)):
            if len(BUFFER) >= chunk_size:
                del BUFFER[-chunk_size:]
            else:
                print(f"finished releasing {INITIAL_BUFFER_SIZE} GB.")
                done_release = True
                break

        sleep_time = 1 - (time.time() - start_time)
        if sleep_time >= 0:
            time.sleep(sleep_time)
        else:
            raise RuntimeError("Received a negative sleep time. The Rate value is too high.")


if __name__ == "__main__":
    task_description = "Dummy program that releases RAM gradually at a given rate."
    rate, chunk_size = extract_rate_and_size(task_description, DEFAULT_CHUNK_SIZE_IN_BYTES)

    # Start with a big allocated buffer
    print(f"Initial allocation: {len(BUFFER)} bytes")
    print(f"Finished allocating at {time.time()}. Starting releasing.")
    release_ram(
        rate=rate,
        chunk_size=chunk_size
    )
