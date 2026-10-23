import argparse
import os
import socket
import subprocess
import signal
import shlex
from typing import Optional, List

import logging

from scanner_trigger import logging_configuration
BUFFER_SIZE = 4192

STOP_MEASUREMENT_MAX_RETRIES = 10
RETRY_INTERVAL_SECONDS = 15

DEFAULT_SCANNER_PATH = r"scanner.py"
DEFAULT_PYTHON_PATH = r"python3"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 65432
DEFAULT_NICE = 0

scanner_process: Optional[subprocess.Popen] = None


def start_measurement(python_path: str, scanner_path: str, nice: int, start_args: List[str]):
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        logging.warning("Got a request to start scanner but scanner is already running, ignoring")
        return

    popen_args = [python_path, scanner_path, *start_args]
    if nice != DEFAULT_NICE:
        popen_args = ["nice", "-n", str(nice)] + popen_args

    scanner_process = subprocess.Popen(popen_args)
    logging.debug(f"Started scanner process, pid = {scanner_process.pid}")


def stop_measurement():
    global scanner_process

    if scanner_process is None:
        logging.warning("Tried to stop a scanner process that does not exist anymore")
        return

    try:
        for retry_num in range(STOP_MEASUREMENT_MAX_RETRIES):
            # It may have exited while we weren't looking.
            return_code = scanner_process.poll()
            if return_code is not None:
                if return_code != 0:
                    logging.warning(f"Scanner exited with error, return code: {return_code}")
                return

            logging.debug(
                f"Sending SIGINT to scanner "
                f"(pid={scanner_process.pid}, "
                f"attempt={retry_num + 1}/{STOP_MEASUREMENT_MAX_RETRIES})"
            )

            if os.name == "nt":
                scanner_process.send_signal(signal.CTRL_C_EVENT)
            else: # POSIX
                scanner_process.send_signal(signal.SIGINT)

            try:
                logging.debug(
                    f"Waiting for scanner to terminate for "
                    f"{RETRY_INTERVAL_SECONDS} seconds "
                    f"(pid={scanner_process.pid})"
                )

                return_code = scanner_process.wait(RETRY_INTERVAL_SECONDS)

                if return_code == 0:
                    logging.info(f"Scanner exited successfully")
                else:
                    logging.warning(f"Scanner exited with error, return code: {return_code}")
                return

            except subprocess.TimeoutExpired:
                logging.warning(
                    f"Scanner did not terminate after "
                    f"{RETRY_INTERVAL_SECONDS} seconds "
                    f"(pid={scanner_process.pid}), "
                    f"retrying ({retry_num + 1}/"
                    f"{STOP_MEASUREMENT_MAX_RETRIES})"
                )

        logging.error(
            f"Scanner did not terminate after "
            f"{STOP_MEASUREMENT_MAX_RETRIES} attempts "
            f"(pid={scanner_process.pid})"
        )

    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt while stopping scanner")
        raise

    finally:
        scanner_process = None


def main(host: str, port: int, python_path: str, scanner_path: str, nice: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logging.info(f"Listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                logging.info(f"Connected to {addr}")
                message = conn.recv(BUFFER_SIZE)
                logging.debug(f"Received a message: {message}")

                if b"start_measurement" in message:
                    start_measurements_args = shlex.split(message.decode())
                    start_measurements_args.remove("start_measurement")
                    start_measurement(python_path, scanner_path, nice, start_measurements_args)

                elif b"stop_measurement" in message:
                    stop_measurement()

                elif b"stop_program" in message:
                    stop_measurement()
                    break


if __name__ == '__main__':
    logging_configuration.setup_logging()
    logging.info("Stating trigger receiver")

    parser = argparse.ArgumentParser(
        description="This script receives a trigger to start and stop the scanner"
    )

    parser.add_argument("-H", "--host",
                        type=str,
                        default=DEFAULT_HOST,
                        help="ip address to listen on")

    parser.add_argument("-P", "--port",
                        type=int,
                        default=DEFAULT_PORT,
                        help="port to listen on")

    parser.add_argument("--python_path",
                        type=str,
                        default=DEFAULT_PYTHON_PATH,
                        help="python path for running the scanner")

    parser.add_argument("--scanner_path",
                        type=str,
                        default=DEFAULT_SCANNER_PATH,
                        help="path to the scanner")

    parser.add_argument("-n", "--nice",
                        type=int,
                        default=DEFAULT_NICE,
                        help="Scanner's priority. Relevant for linux only")

    args = parser.parse_args()

    main(args.host, args.port, args.python_path, args.scanner_path, args.nice)
