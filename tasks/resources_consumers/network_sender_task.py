import os
import socket
import time
import psutil

from tasks.resources_consumers.task_utils import extract_rate_and_size

UDP_IP = "192.168.1.117"
UDP_PORT = 12345
PACKET_SIZE = 1024


def send_udp_packets(rate: float, packet_size: int = PACKET_SIZE, ip: str = UDP_IP, port: int = UDP_PORT):
    """
    Sends UDP packets endlessly at a specified rate (packets/sec) and packet size.
    Stops only when the program is terminated externally.
    """
    p = psutil.Process()
    print("Before sending:", p.io_counters())
    print(psutil.net_io_counters())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / rate  # seconds between packets

    try:
        while True:
            message = os.urandom(packet_size)
            sock.sendto(message, (ip, port))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping packet sending...")
    finally:
        sock.close()
        print("After sending:", p.io_counters())
        print(psutil.net_io_counters())


if __name__ == "__main__":
    task_description = "Sends UDP packets at a given rate and packet size endlessly until stopped."
    rate, packet_size = extract_rate_and_size(task_description, PACKET_SIZE)

    send_udp_packets(
        rate=rate,
        packet_size=packet_size
    )
