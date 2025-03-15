import argparse
import logging
import socket
from typing import List, Tuple

from scanner_trigger import logging_configuration

DEFAULT_TRIGGER_RECEIVER_HOST = "127.0.0.1"
DEFAULT_TRIGGER_RECEIVER_PORT = 65432
CONNECT_TIMEOUT = 5


def parse_addresses(value: str) -> List[Tuple[str, int]]:
    addresses = value.split(',')
    result = []
    for address in addresses:
        ip, port = address.split(':')
        result.append((ip.strip(), int(port.strip())))
    return result


def decorate_addresses(addresses: List[Tuple[str, int]]) -> str:
    return "\n".join([f"Host: {host}, Port: {port}" for host, port in addresses])


def main(trigger_message: str, receivers_addresses: List[Tuple[str, int]]) -> None:
    logging.info("Sending trigger to the following addresses:\n" + decorate_addresses(receivers_addresses))

    for receiver_address in receivers_addresses:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(CONNECT_TIMEOUT)
            try:
                s.connect(receiver_address)
            except socket.timeout:
                logging.warning(f"Connect timout ({CONNECT_TIMEOUT} seconds). Address: {receiver_address}, ")
                continue
            logging.info(f"Connected to {receiver_address}, Sending:{trigger_message}")
            s.sendall(trigger_message.encode())


if __name__ == '__main__':
    logging_configuration.setup_logging()
    logging.info("Stating trigger sender")

    parser = argparse.ArgumentParser(
        description="""This script sends a trigger to the scanner trigger receivers. 
Note: this script is synchronous and could be further improved by sending triggers asynchronously"""
    )

    parser.add_argument('trigger_message',
                        type=str,
                        choices=['start_measurement', 'stop_measurement', 'stop_program'],
                        help="The trigger message")
    parser.add_argument('-r', '--receivers_addresses',
                        type=parse_addresses,
                        help="""The receivers addresses to send the trigger to.
Example: 1.1.1.1:80,2.2.2.2:90,3.3.3.3:100""",
                        default=[(DEFAULT_TRIGGER_RECEIVER_HOST, DEFAULT_TRIGGER_RECEIVER_PORT)])

    args = parser.parse_args()

    main(args.trigger_message, args.receivers_addresses)
