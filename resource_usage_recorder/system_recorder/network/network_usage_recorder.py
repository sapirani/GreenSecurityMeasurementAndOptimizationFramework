import psutil

from DTOs.scanner_counters.system.network_results import SystemNetworkResults
from utils.general_consts import KB
from resource_usage_recorder import MetricRecorder


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
