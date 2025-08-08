import argparse
import time

def consume_ram(speed: float, chunk_size: int, running_time: float):
    global_start_time = time.time()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only consumes RAM"
    )

    parser.add_argument("-t", "--duration",
                        type=int,
                        required=True,
                        help="The duration of executing the task")

    parser.add_argument("-s", "--speed",
                        type=float,
                        required=True,
                        help="The speed (bytes per second) of consumption")

    parser.add_argument("-c", "--chunk_size",
                        type=int,
                        required=True,
                        help="The chunk size (in bytes) of consumption.")

    args = parser.parse_args()
    consume_ram(speed=args.speed, chunk_size=args.chunk_size, running_time=args.duration)

