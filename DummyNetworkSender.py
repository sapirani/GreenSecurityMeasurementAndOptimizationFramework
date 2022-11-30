import socket

import psutil

UDP_IP = "192.1.1.1"
UDP_PORT = 5005
MESSAGE = "Hello, World!"


p = psutil.Process()
print(p.io_counters())
print(psutil.net_io_counters())
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
for i in range(1000000):
    sock.sendto(bytes(MESSAGE + f" {i}", "utf-8"), (UDP_IP, UDP_PORT))
    #time.sleep(0.00000000001)

print(p.io_counters())
print(psutil.net_io_counters())
