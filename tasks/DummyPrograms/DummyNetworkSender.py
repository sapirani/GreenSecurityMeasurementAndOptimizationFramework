import socket
import time
import psutil

UDP_IP = "127.0.0.1"
UDP_PORT = 12345
MESSAGE = b"Hello, World!"
DEFAULT_TIME_INTERVAL = 0.01


def send_udp_packets(duration: int, ip: str = UDP_IP, port: int = UDP_PORT, interval: int = DEFAULT_TIME_INTERVAL):
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
    send_udp_packets(duration=60)  # Run for 10 seconds