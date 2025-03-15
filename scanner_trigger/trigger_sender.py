import argparse
import logging
import socket

from scanner_trigger import logging_configuration

TRIGGER_RECEIVER_HOST = "172.21.41.238"
TRIGGER_RECEIVER_PORT = 65432


def main(trigger_message: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((TRIGGER_RECEIVER_HOST, TRIGGER_RECEIVER_PORT))
        logging.info(f"Connected to {TRIGGER_RECEIVER_HOST}:{TRIGGER_RECEIVER_PORT}, Sending:{trigger_message}")
        s.sendall(trigger_message.encode())


if __name__ == '__main__':
    logging_configuration.setup_logging()
    parser = argparse.ArgumentParser(
        description="This script sends a trigger to the scanner trigger receivers"
    )

    parser.add_argument('trigger_message',
                        type=str,
                        choices=['start_measurement', 'stop_measurement', 'stop_program'],
                        help="The trigger message")

    args = parser.parse_args()

    main(args.trigger_message)
