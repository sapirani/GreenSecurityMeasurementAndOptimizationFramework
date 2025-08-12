import time

from tasks.resources_consumers.task_utils import extract_rate_and_size
from utils.general_consts import GB

CHUNK_SIZE = 2 ** 10  # 1 KB
INITIAL_BUFFER_SIZE = 5 * GB
INITIAL_BUFFER = bytearray(INITIAL_BUFFER_SIZE)


def release_ram(rate: float, chunk_size: int = CHUNK_SIZE):
    """
    Gradually releases memory from a pre-allocated buffer
    at the given rate (bytes/sec) and chunk size.
    """

    while True:
        start_time = time.time()

        # Determine how many chunks to release this cycle
        chunks_to_release = int(rate / chunk_size)

        for _ in range(chunks_to_release):
            if len(INITIAL_BUFFER) >= chunk_size:
                del INITIAL_BUFFER[-chunk_size:]
            else:
                print(f"finished releasing {INITIAL_BUFFER_SIZE} GB.")

        sleep_time = 1 - (time.time() - start_time)
        if sleep_time >= 0:
            time.sleep(sleep_time)
        else:
            raise RuntimeError("Received a negative sleep time. The Rate value is too high.")


if __name__ == "__main__":
    task_description = "Dummy program that releases RAM gradually at a given rate."
    rate, chunk_size = extract_rate_and_size(task_description, CHUNK_SIZE)

    # Start with a big allocated buffer
    print(f"Initial allocation: {len(INITIAL_BUFFER)} bytes")
    print(f"Finished allocating at {time.time()}. Starting releasing.")
    release_ram(
        rate=rate,
        chunk_size=chunk_size
    )
