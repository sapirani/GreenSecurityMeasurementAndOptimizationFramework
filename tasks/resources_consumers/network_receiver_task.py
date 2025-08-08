import argparse
import socket
UDP_IP = "192.168.1.145"
UDP_PORT = 12345
TIMEOUT = 2.0
BUFFER_SIZE = 1024

def receive_udp_packets(ip: int = UDP_IP, port: int = UDP_PORT, buffer_size: int = BUFFER_SIZE):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"Listening for UDP packets on {ip}:{port}...")

    while True:
        try:
            data, addr = sock.recvfrom(buffer_size)
            print(f"Received packet from {addr}: {data}")
        except socket.timeout:
            continue

    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only receives packets through network"
    )

    parser.add_argument("-a", "--ip_address",
                        type=str,
                        required=True,
                        help="The ip address of the device that receives the messages.")

    parser.add_argument("-p", "--port",
                        type=int,
                        required=True,
                        help="The port on which the device is listening.")

    parser.add_argument("-s", "--buffer_size",
                        type=int,
                        default=BUFFER_SIZE,
                        help="The size of the message.")

    args = parser.parse_args()

    receive_udp_packets(ip=args.ip_address, port=args.port, buffer_size=args.buffer_size)