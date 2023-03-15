from scapy.all import *
from scapy.layers.inet import IP, ICMP, TCP

#send(IP(dst="1.1.2.2", src="1.5.1.6")/ICMP())
from scapy.layers.l2 import Ether

pac = Ether(src='D8:D0:90:23:3F:58', dst='08:92:04:84:6E:62')/IP(dst="1.1.2.2", src="1.5.1.6")/TCP()
sendp(pac)
print(pac)