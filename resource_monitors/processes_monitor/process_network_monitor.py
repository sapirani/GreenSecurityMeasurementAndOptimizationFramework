import dataclasses
from collections import defaultdict
from threading import Thread, Lock
import psutil
from scapy.all import ifaces, packet
from scapy.arch import get_if_addr
from scapy.interfaces import get_working_ifaces
from scapy.sendrecv import sniff


@dataclasses.dataclass
class NetworkStats:
    bytes_sent: int = 0
    packets_sent: int = 0
    bytes_received: int = 0
    packets_received: int = 0


class ProcessNetworkMonitor:
    """
    This class is responsible for monitoring the number of packets and the total size sent and received from this device
    Disclaimer: it may be difficult to capture short outgoing sessions, since the connections of the process are not
    monitored frequently enough, so the mapping between local port to pid will be missing so that packet will be ignored
    This is not the case for incoming sessions since we have a listening socket for the process.

    Note: this class monitors packets captured in any interface
    """
    def __init__(self, interfaces):
        self._interfaces = interfaces

        self.pid2traffic = defaultdict(lambda: NetworkStats())
        self.pid2traffic_lock = Lock()

        self.local_port_to_pid = {}
        self.device_mac_addresses = {iface.mac for iface in ifaces.values()}
        self.all_ips = {get_if_addr(iface) for iface in get_working_ifaces()}

        self._stop_sniffing = False

        self.sniffing_thread = Thread(target=self._sniff_packets)

    def start(self):
        self.sniffing_thread.start()

    def _sniff_packets(self):
        """
        Sniff packets until receiving some trigger to stop (self._stop_sniffing:).
        Note: if we are not receiving any packet, without the timeout flag we will bw stuck here forever.
        (since the stop_filter is examined upon receiving a packet)
        Hence, we force timeout to ensure that self._stop_sniffing is still False.
        """
        try:
            while not self._stop_sniffing:
                sniff(prn=self._process_packet, iface=self._interfaces,
                      store=False, timeout=3, stop_filter=lambda _: self._stop_sniffing)
        except PermissionError:
            print("warning! per-process network measurements require elevations")

    def _is_outgoing_packet(self, captured_packet):
        return captured_packet.src in self.device_mac_addresses or captured_packet.src in self.all_ips

    def _identify_local_port(self, source_port, destination_port, captured_packet):
        return source_port if self._is_outgoing_packet(captured_packet) else destination_port

    def _update_pid2traffic_stats(self, captured_packet: packet, packet_pid):
        if self._is_outgoing_packet(captured_packet):
            # source MAC address is ours - this packet is being sent
            with self.pid2traffic_lock:
                self.pid2traffic[packet_pid].bytes_sent += len(captured_packet)
                self.pid2traffic[packet_pid].packets_sent += 1

        else:
            # destination MAC address is ours - this packet is being received
            with self.pid2traffic_lock:
                self.pid2traffic[packet_pid].bytes_received += len(captured_packet)
                self.pid2traffic[packet_pid].packets_received += 1

    def _update_all_local_port_to_pid(self):
        for c in psutil.net_connections():
            if c.laddr:
                self.local_port_to_pid[c.laddr.port] = c.pid

    def _process_packet(self, captured_packet: packet):
        try:
            source_port, destination_port = captured_packet.sport, captured_packet.dport
        except (AttributeError, IndexError):
            # Ignore packets without transport layers
            pass
        else:
            local_port = self._identify_local_port(source_port, destination_port, captured_packet)
            packet_pid = self.local_port_to_pid.get(local_port)
            if packet_pid:
                self._update_pid2traffic_stats(captured_packet, packet_pid)
            else:
                self._update_all_local_port_to_pid()
                packet_pid = self.local_port_to_pid.get(local_port)
                if packet_pid:    # otherwise, ignore the packet
                    self._update_pid2traffic_stats(captured_packet, packet_pid)

    def _update_process_connections(self, pid, process_connections):
        for connection in process_connections:
            if connection.laddr:
                self.local_port_to_pid[connection.laddr.port] = pid

    def get_network_stats(self, process: psutil.Process) -> NetworkStats:
        try:
            process_connections = process.net_connections()
        except psutil.NoSuchProcess:
            return NetworkStats()   # all fields are 0 by default

        self._update_process_connections(process.pid, process_connections)

        with self.pid2traffic_lock:
            traffic = self.pid2traffic[process.pid]
            self.pid2traffic[process.pid] = NetworkStats()

        return traffic

    def stop(self):
        self._stop_sniffing = True
        self.sniffing_thread.join()
