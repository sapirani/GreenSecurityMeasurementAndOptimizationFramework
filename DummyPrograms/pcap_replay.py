from scapy.all import *
import time
import sys
import platform
import psutil

from threading import Timer

SPEED_LEVEL = 9     # 0 will send the packets with no sleep at all. 9 will send the packets in the lowest speed.
TO_PRINT_AFTER_NUMER_OF_PACKETS = 500

max_packets_per_second = 20  # sending 20 packets every second
number_of_levels = 10
added_packets_in_each_level = max_packets_per_second / number_of_levels
transmission_rate = 1 / max_packets_per_second

sleep_times = [(transmission_rate * level) / (max_packets_per_second - level * added_packets_in_each_level)
               for level in range(number_of_levels)]

MINUTE = 60
TIME_LIMIT = 1 * MINUTE
SLEEP_TIME_BETWEEN_PACKETS = sleep_times[SPEED_LEVEL]

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


def terminate_program(p):
    global packet_counter
    print("=====================================")
    print("     total packets sent:", packet_counter + 1)
    print("=====================================")
    p.terminate()


t = Timer(TIME_LIMIT, terminate_program, [psutil.Process()])
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

        if (packet_counter + 1) % TO_PRINT_AFTER_NUMER_OF_PACKETS == 0:
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
