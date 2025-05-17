"""arr = [1] * 10000

while True:
    arr.append(2)"""
import sys
import time


def consume_ram(speed):
    global_start_time = time.time()
    chunk_size = int(sys.argv[1])
    running_time = int(sys.argv[3])
    arr = b''
    while True:
        start_time = time.time()
        for _ in range(int(speed / chunk_size)):
            arr += bytearray(chunk_size)

        sleep_time = 1 - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        if time.time() - global_start_time > running_time:
            return


# First argument - chunk size
# Second argument - speed (bytes per second)
# Third argument - running time
consume_ram(float(sys.argv[2]))

    