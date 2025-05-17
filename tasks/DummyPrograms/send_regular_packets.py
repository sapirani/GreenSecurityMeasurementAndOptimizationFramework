from scapy.all import *
from scapy.layers.inet import IP, ICMP, TCP

send(IP(dst='192.168.1.100', src='1.1.12.1')/ICMP())
#sendp(IP(dst='www.google.com') / TCP(dport=80, flags='S'))
"""for i in range(50):
    packet = IP(dst="192.168.100.123")/TCP()/"from scapy packet"
    send(packet)"""