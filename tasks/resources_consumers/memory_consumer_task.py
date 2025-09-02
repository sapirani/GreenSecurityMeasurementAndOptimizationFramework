import time

from tasks.resources_consumers.task_utils import extract_rate_and_size

CHUNK_SIZE = 2 ** 10


def consume_ram(rate: float, chunk_size: int = CHUNK_SIZE):
    arr = b''
    interval = 1.0 / rate
    while True:
        start_time = time.time()
        for _ in range(int(rate)):
            arr += bytearray(chunk_size)

        elapsed = time.time() - start_time
        sleep_time = interval - elapsed
        if sleep_time >= 0:
            time.sleep(sleep_time)
        else:
            raise RuntimeError("Received a negative sleep time. The Rate value is too high.")


if __name__ == "__main__":
    task_description = "This program is a dummy task that only consumes RAM"
    rate, chunk_size = extract_rate_and_size(task_description, CHUNK_SIZE)

    consume_ram(
        rate=rate,
        chunk_size=chunk_size
    )
