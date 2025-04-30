import socket
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
TIMEOUT = 2.0
BUFFER_SIZE = 1024

def receive_udp_packets(ip: int = UDP_IP, port: int = UDP_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"Listening for UDP packets on {ip}:{port}...")

    sock.settimeout(TIMEOUT)  # Avoid blocking forever

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            print(f"Received packet from {addr}: {data}")
        except socket.timeout:
            continue

    sock.close()


if __name__ == "__main__":
    receive_udp_packets()