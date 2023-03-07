from scapy.all import *
import time
import sys

packets = rdpcap(sys.argv[1])
clk = float(packets[0].time)
for index, p in enumerate(packets):
    next_time = float(p.time)
    time.sleep(next_time - clk)
    clk = next_time
    sendp(p, verbose=False)

    if (index + 1) % 500 == 0:
        print(f"sent {index + 1} packets (in total)")

