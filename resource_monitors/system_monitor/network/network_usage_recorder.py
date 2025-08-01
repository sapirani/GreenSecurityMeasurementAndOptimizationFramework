import dataclasses

import psutil
from utils.general_consts import KB
from resource_monitors import MetricResult, MetricRecorder


@dataclasses.dataclass
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


class SystemNetworkUsageRecorder(MetricRecorder):
    def __init__(self):
        self._previous_network_io = self._get_total_results()

    @staticmethod
    def _get_total_results() -> SystemNetworkResults:
        network_io_counters = psutil.net_io_counters()
        return SystemNetworkResults(
            packets_sent=network_io_counters.packets_sent,
            packets_received=network_io_counters.packets_recv,
            network_kb_sent=network_io_counters.bytes_sent / KB,
            network_kb_received=network_io_counters.bytes_recv / KB,
        )

    def get_current_metrics(self) -> SystemNetworkResults:
        total_results = self._get_total_results()
        current_results = total_results.delta(self._previous_network_io)
        self._previous_network_io = total_results
        return current_results

