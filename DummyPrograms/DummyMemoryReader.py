import ctypes
import sys
import time


def read_memory_chunks(speed_kb_per_sec):
    chunk_size = 1024  # 1 kilobyte
    duration = chunk_size / (speed_kb_per_sec * 1024)

    # Allocate memory using ctypes
    buffer = (ctypes.c_ubyte * chunk_size)()

    # Read memory chunks without cache optimizations
    while True:
        # Access each byte in the buffer
        for i in range(chunk_size):
            _ = buffer[i]
            # Do something with the byte (print, process, etc.)
        time.sleep(duration)


read_memory_chunks(100)

chunk_size = sys.argv[1]
speed = sys.argv[2]
