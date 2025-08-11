import argparse
import os
import socket
import sys
import time
import psutil

UDP_IP = "192.168.1.117"
UDP_PORT = 12345
DEFAULT_TIME_INTERVAL = 0.01
PACKET_SIZE = 1024

DURATION_PARAM_INDEX = 1
INTERVAL_PARAM_INDEX = 2



def send_udp_packets(duration: float, ip: str = UDP_IP, port: int = UDP_PORT, interval: float = DEFAULT_TIME_INTERVAL, packet_size: int = PACKET_SIZE):
    p = psutil.Process()
    print(p.io_counters())
    print(psutil.net_io_counters())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    end_time = time.time() + duration

    while time.time() < end_time:
        message = os.urandom(packet_size)
        sock.sendto(message, (ip, port))
        time.sleep(interval)

    sock.close()
    print(p.io_counters())
    print(psutil.net_io_counters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only receives packets through network"
    )

    parser.add_argument("-a", "--ip_address",
                        type=str,
                        required=True,
                        help="The ip address of the device that sends the messages.")

    parser.add_argument("-p", "--port",
                        type=int,
                        required=True,
                        help="The port on which the device is listening.")

    parser.add_argument("-s", "--packet_size",
                        type=int,
                        default=PACKET_SIZE,
                        help="The size of the packet sent.")


    parser.add_argument("-t", "--duration",
                        type=float,
                        required=True,
                        help="The duration of the task.")


    parser.add_argument("-i", "--time_interval",
                        type=float,
                        required=True,
                        help="The time interval between two packets.")
    args = parser.parse_args()

    send_udp_packets(duration=args.duration, ip=args.ip_address, port=args.port, interval=args.time_interval, packet_size=args.packet_size)