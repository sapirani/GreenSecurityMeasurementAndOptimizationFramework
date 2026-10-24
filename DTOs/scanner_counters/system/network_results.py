from dataclasses import dataclass

from DTOs.scanner_counters.metric_results import MetricResult


@dataclass
class SystemNetworkResults(MetricResult):
    packets_sent: int
    packets_received: int
    network_kb_sent: float
    network_kb_received: float

    def delta(self, previous_disk_io: 'SystemNetworkResults') -> 'SystemNetworkResults':
        return SystemNetworkResults(
            packets_sent=self.packets_sent - previous_disk_io.packets_sent,
            packets_received=self.packets_received - previous_disk_io.packets_received,
            network_kb_sent=round(self.network_kb_sent - previous_disk_io.network_kb_sent, 3),
            network_kb_received=round(self.network_kb_received - previous_disk_io.network_kb_received, 3)
        )
