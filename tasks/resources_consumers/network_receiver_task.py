import socket
import time

from tasks.resources_consumers.task_utils import extract_rate_and_size

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345
TIMEOUT = 2.0
DEFAULT_BUFFER_SIZE_IN_BYTES = 1024


def receive_udp_packets(rate: float, buffer_size: int = DEFAULT_BUFFER_SIZE_IN_BYTES, ip: str = UDP_IP, port: int = UDP_PORT):
    """
    Receives UDP packets endlessly at a specified read rate (packets/sec)
    and packet size (buffer size).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    sock.settimeout(TIMEOUT)

    interval = 1.0 / rate  # expected processing interval

    print(f"Listening for UDP packets on {ip}:{port} (rate: {rate} pps, packet_size: {buffer_size} bytes)...")

    try:
        while True:
            start_time = time.time()
            try:
                data, addr = sock.recvfrom(buffer_size)
                print(f"Received packet from {addr}: {len(data)} bytes")
            except socket.timeout:
                pass

            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time >= 0:
                time.sleep(sleep_time)
            else:
                raise RuntimeError("Received a negative sleep time. The Rate value is too high.")
    except KeyboardInterrupt:
        print("\nStopping packet receiver...")
    finally:
        sock.close()


if __name__ == "__main__":
    task_description = "Receives UDP packets endlessly at a given processing rate and packet size."
    rate, buffer_size = extract_rate_and_size(task_description, DEFAULT_BUFFER_SIZE_IN_BYTES)

    receive_udp_packets(
        rate=rate,
        buffer_size=buffer_size
    )
