import time
import argparse

from tasks.resources_consumers.task_utils import extract_rate_and_size

CHUNK_SIZE = 2 ** 10  # 1 KB


def release_ram(rate: float, chunk_size: int = CHUNK_SIZE):
    """
    Gradually releases memory from a pre-allocated buffer
    at the given rate (bytes/sec) and chunk size.
    """
    # Start with a big allocated buffer
    arr = bytearray(int(rate * 5))  # start with ~5 seconds worth of memory
    print(f"Initial allocation: {len(arr)} bytes")

    while True:
        start_time = time.time()

        # Determine how many chunks to release this cycle
        chunks_to_release = int(rate / chunk_size)

        for _ in range(chunks_to_release):
            if len(arr) >= chunk_size:
                arr = arr[:-chunk_size]  # remove last chunk
            else:
                # Nothing left to release, reallocate a big buffer for continuous release
                arr = bytearray(int(rate * 5))
                print("Buffer refilled for releasing again")

        sleep_time = 1 - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    task_description = "Dummy program that releases RAM gradually at a given rate."
    rate, chunk_size = extract_rate_and_size(task_description, CHUNK_SIZE)

    release_ram(
        rate=rate,
        chunk_size=chunk_size
    )
