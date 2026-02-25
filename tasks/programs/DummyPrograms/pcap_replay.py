from scapy.all import *
import time
import sys
import platform
import psutil

from threading import Timer

MINUTE = 60

########## Parameters To Change ##########
TIME_LIMIT = 20
SPEED_LEVEL = 0  # 0 will send the packets with no sleep at all. 9 will send the packets in the lowest speed.
TO_PRINT_AFTER_NUMER_OF_PACKETS = 500
#########################################


max_packets_per_second = 23.33333333  # sending 20 packets every second
number_of_levels = 10
added_packets_in_each_level = max_packets_per_second / number_of_levels
transmission_rate = 1 / max_packets_per_second

sleep_times = [(transmission_rate * level * added_packets_in_each_level) / (max_packets_per_second - level * added_packets_in_each_level)
               for level in range(number_of_levels)]


SLEEP_TIME_BETWEEN_PACKETS = sleep_times[SPEED_LEVEL]

INTERFACE_NAME = "enx089204846f61"

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
big_packets_counter = 0


def print_summary():
    global big_packets_counter
    global packet_counter
    print("=====================================")
    print("     total of packets sniffed:", packet_counter + 1)
    print("     total of too big packets:", big_packets_counter + 1)
    print("=====================================")


def terminate_program(p):
    print_summary()
    p.terminate()


t = Timer(TIME_LIMIT, terminate_program, [psutil.Process()])
t.start()

print("start")


def send_packets(p):
    global packet_counter
    global first_time
    global first_pcap_time
    global big_packets_counter

    if len(p) <= mtu:
        sendp(p, verbose=False)
        time.sleep(SLEEP_TIME_BETWEEN_PACKETS)

        if (packet_counter + 1) % TO_PRINT_AFTER_NUMER_OF_PACKETS == 0:
            print(f"sent {packet_counter + 1} packets (in total)")

    else:
        big_packets_counter += 1
        print(f"message is too long: {len(p)} bytes. Index: {packet_counter}")
    packet_counter += 1


sniff(offline=sys.argv[1], prn=send_packets)
t.cancel()
print_summary()


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
