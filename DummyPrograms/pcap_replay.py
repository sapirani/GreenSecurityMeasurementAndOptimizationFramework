from scapy.all import *
import time
import sys
import platform

INTERFACE_NAME = "wlp0s20f3"

if platform.system() == "Linux":
    res = subprocess.run(f'cat /sys/class/net/{INTERFACE_NAME}/mtu', capture_output=True, shell=True)
    if res.returncode != 0:
        raise Exception("cannot get the value of MTU", res.stderr)

    mtu = int(res.stdout.decode("utf-8"))

else:
    mtu = 1500


first_time = None
first_pcap_time = None
packet_counter = 0


def send_packets(p):
    global packet_counter
    global first_time
    global first_pcap_time

    if len(p) <= mtu:
        next_time = float(p.time)

        if first_time is None:
            first_time = time.time()
            first_pcap_time = next_time
            print("start")

        wait_time = (next_time - first_pcap_time) - (time.time() - first_time)
        if wait_time > 0:
            time.sleep(wait_time)
        sendp(p, verbose=False)

        if (packet_counter + 1) % 10 == 0:
            print(f"sent {packet_counter + 1} packets (in total)")

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

