from scapy.all import *
import time
import sys

INTERFACE_NAME = ""

res = subprocess.run(f'cat /sys/class/net/{INTERFACE_NAME}/mtu', capture_output=True, shell=True)
if res.returncode != 0:
    raise Exception("cannot get the value of MTU", res.stderr)

mtu = int(res.stdout.decode("utf-8"))
print(mtu)

packet_counter = 0


def send_packets(p):
    global packet_counter
    #print(p, packet_counter)
    if len(p) <= mtu:
        sendp(p)
    else:
        print(f"message is too long: {len(p)} bytes. Index: {packet_counter}")
    packet_counter += 1


sniff(offline=sys.argv[1], prn=send_packets)

"""packets = rdpcap(sys.argv[1])
clk = float(packets[0].time)
for index, p in enumerate(packets):
    next_time = float(p.time)
    time.sleep(next_time - clk)
    clk = next_time
    sendp(p, verbose=False)

    if (index + 1) % 500 == 0:
        print(f"sent {index + 1} packets (in total)")"""

