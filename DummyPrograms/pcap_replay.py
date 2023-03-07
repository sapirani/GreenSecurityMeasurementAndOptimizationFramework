from scapy.all import *
import time
import sys

packet_counter = 0


def send_packets(p):
    global packet_counter
    #print(p, packet_counter)
    try:
        sendp(p)
    except OSError:
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

