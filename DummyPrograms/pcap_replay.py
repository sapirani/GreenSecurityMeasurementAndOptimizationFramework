from scapy.all import *
import time
import sys
import platform
import psutil

from threading import Timer

sleep_times = [i / 20 for i in range(10)]

from scapy.layers.inet import IP, ICMP, TCP

from scapy.layers.inet import ICMP
MINUTE = 60
TIME_LIMIT = 1 * MINUTE
SLEEP_TIME_BETWEEN_PACKETS = sleep_times[1]   # 0 will send the packets with no sleep at all. 9 will send the packets in the lowest speed.

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


t = Timer(TIME_LIMIT, lambda p: p.terminate(), [psutil.Process()])
t.start()

print("start")


def send_packets(p):
    global packet_counter
    global first_time
    global first_pcap_time

    if len(p) <= mtu:
        """next_time = float(p.time)

        if first_time is None:
            first_time = time.time()
            first_pcap_time = next_time
            print("start")

        wait_time = (next_time - first_pcap_time) - (time.time() - first_time)
        if wait_time > 0:
            time.sleep(wait_time)"""
        time.sleep(SLEEP_TIME_BETWEEN_PACKETS)
        sendp(p, verbose=False)
        #send(IP(src='172.16.3.10', dst='1.1.12.1') / ICMP())
        #send(IP(dst='www.google.com') / TCP(dport=80, flags='S'))

        if (packet_counter + 1) % 500 == 0:
            print(f"sent {packet_counter + 1} packets (in total)")

    else:
        print(f"message is too long: {len(p)} bytes. Index: {packet_counter}")
    packet_counter += 1


sniff(offline=sys.argv[1], prn=send_packets)
t.cancel()
"""packets = rdpcap(sys.argv[1])
clk = float(packets[0].time)
for index, p in enumerate(packets):
    next_time = float(p.time)
    time.sleep(next_time - clk)
    clk = next_time
    #p[IP].src = '172.16.54.240'
    send(p.payload, verbose=False)
    print(p.payload)

    if (index + 1) % 10 == 0:
        print(f"sent {index + 1} packets (in total)")"""
