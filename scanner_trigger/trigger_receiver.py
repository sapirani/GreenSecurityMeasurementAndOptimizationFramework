import argparse
import os
import socket
import subprocess
import signal
from typing import Optional

import logging

from scanner_trigger import logging_configuration

SCANNER_TERMINATION_WAITING_SECONDS = 30

scanner_path = r"scanner.py"
python_path = r"python3"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 65432

scanner_process: Optional[subprocess.Popen] = None


def start_measurement() -> None:
    global scanner_process
    if scanner_process:
        logging.warning("Got a request to start scanner but scanner is already running, ignoring")
        return
    
    scanner_process = subprocess.Popen([python_path, scanner_path])


def stop_measurement() -> None:
    global scanner_process

    if not scanner_process:
        logging.warning("Tried to stop a scanner process that does not exist anymore")
        return

    if os.name == 'nt':  # Check if the OS is Windows
        logging.debug("Sending Windows style SIGNIT to the scanner")
        scanner_process.send_signal(signal.CTRL_C_EVENT)  # CTRL_C_EVENT on Windows
    else:
        logging.debug("Sending LINUX style SIGNIT to the scanner")
        scanner_process.send_signal(signal.SIGINT)  # SIGINT on UNIX-like systems

    try:
        logging.debug(f"Waiting for scanner to terminate for {SCANNER_TERMINATION_WAITING_SECONDS} seconds")
        scanner_process.wait(SCANNER_TERMINATION_WAITING_SECONDS)
    except subprocess.TimeoutExpired:
        logging.warning(f"Scanner did not terminate after {SCANNER_TERMINATION_WAITING_SECONDS} seconds, skipping ...")
    except KeyboardInterrupt:
        pass

    scanner_process = None


def main(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logging.info(f"Listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                logging.info(f"Connected to {addr}")
                message = conn.recv(64)
                logging.debug(f"Received a message: {message}")

                if message == b"start_measurement":
                    start_measurement()

                elif message == b"stop_measurement":
                    stop_measurement()

                elif message == b"stop_program":
                    stop_measurement()
                    break


if __name__ == '__main__':
    logging_configuration.setup_logging()

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

    args = parser.parse_args()

    main(args.host, args.port)
