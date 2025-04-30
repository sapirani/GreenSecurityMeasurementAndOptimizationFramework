import socket
import sys
import time
import psutil

UDP_IP = "127.0.0.1"
UDP_PORT = 12345
MESSAGE = b"Hello, World!"
DEFAULT_TIME_INTERVAL = 0.01

DURATION_PARAM_INDEX = 1
INTERVAL_PARAM_INDEX = 2


def send_udp_packets(duration: float, ip: str = UDP_IP, port: int = UDP_PORT, interval: float = DEFAULT_TIME_INTERVAL):
    p = psutil.Process()
    print(p.io_counters())
    print(psutil.net_io_counters())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    end_time = time.time() + duration

    while time.time() < end_time:
        sock.sendto(MESSAGE, (ip, port))
        time.sleep(interval)

    sock.close()
    print(p.io_counters())
    print(psutil.net_io_counters())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Usage: python DummyNetworkSender.py <duration>")

    duration = float(sys.argv[DURATION_PARAM_INDEX])
    if len(sys.argv) >= 3:
        time_interval = float(sys.argv[INTERVAL_PARAM_INDEX])
        send_udp_packets(duration=duration, interval=time_interval)

    send_udp_packets(duration=duration)