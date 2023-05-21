"""arr = [1] * 10000

while True:
    arr.append(2)"""


import time


def consume_ram(speed_mb_per_sec):
    chunk_size = 1024 * 1024  # 1 MB
    duration = chunk_size / (speed_mb_per_sec * 1024 * 1024)

    while True:
        _ = bytearray(chunk_size)
        time.sleep(duration)


# Example usage: consuming RAM at a speed of 100 MB/s
consume_ram(1)

    