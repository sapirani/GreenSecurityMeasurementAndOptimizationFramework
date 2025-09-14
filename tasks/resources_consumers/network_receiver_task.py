import argparse
import socket
import time

from tasks.resources_consumers.task_utils import extract_rate_and_size

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345
TIMEOUT = 2.0
DEFAULT_BUFFER_SIZE_IN_BYTES = 1024


def receive_udp_packets(buffer_size: int = DEFAULT_BUFFER_SIZE_IN_BYTES, ip: str = UDP_IP, port: int = UDP_PORT):
    """
    Receives UDP packets endlessly at a specified read rate (packets/sec)
    and packet size (buffer size).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    sock.settimeout(TIMEOUT)


    print(f"Listening for UDP packets on {ip}:{port} (packet_size: {buffer_size} bytes)...")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(buffer_size)
                print(f"Received packet from {addr}: {len(data)} bytes")
            except socket.timeout:
                pass

    except KeyboardInterrupt:
        print("\nStopping packet receiver...")
    finally:
        sock.close()


if __name__ == "__main__":
    task_description = "Receives UDP packets endlessly at a given processing rate and packet size."
    parser = argparse.ArgumentParser(description=task_description)

    parser.add_argument("-s", "--buffer_size",
                        type=int,
                        default=DEFAULT_BUFFER_SIZE_IN_BYTES,
                        help="The size of the buffer in bytes.")


    receive_udp_packets(
        buffer_size=parser.parse_args().buffer_size
    )
