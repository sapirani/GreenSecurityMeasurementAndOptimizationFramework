import dataclasses

import psutil
from utils.general_consts import KB
from resource_monitors import MetricResult, MetricRecorder


@dataclasses.dataclass
class SystemDiskResults(MetricResult):
    disk_read_count: int
    disk_write_count: int
    disk_read_kb: float
    disk_write_kb: float
    disk_read_time: int
    disk_write_time: int

    def delta(self, previous_disk_io: 'SystemDiskResults') -> 'SystemDiskResults':
        return SystemDiskResults(
            disk_read_count=self.disk_read_count - previous_disk_io.disk_read_count,
            disk_write_count=self.disk_write_count - previous_disk_io.disk_write_count,
            disk_read_kb=round(self.disk_read_kb - previous_disk_io.disk_read_kb, 3),
            disk_write_kb=round(self.disk_write_kb - previous_disk_io.disk_write_kb, 3),
            disk_read_time=self.disk_read_time - previous_disk_io.disk_read_time,
            disk_write_time=self.disk_write_time - previous_disk_io.disk_write_time
        )


class SystemDiskUsageRecorder(MetricRecorder):
    def __init__(self):
        self._previous_disk_io = self._get_total_results()

    @staticmethod
    def _get_total_results() -> SystemDiskResults:
        disk_io_counters = psutil.disk_io_counters()
        return SystemDiskResults(
            disk_read_count=disk_io_counters.read_count,
            disk_write_count=disk_io_counters.write_count,
            disk_read_kb=disk_io_counters.read_bytes / KB,
            disk_write_kb=disk_io_counters.write_bytes / KB,
            disk_read_time=disk_io_counters.read_time,
            disk_write_time=disk_io_counters.write_time,
        )

    def get_current_metrics(self) -> SystemDiskResults:
        total_results = self._get_total_results()
        current_results = total_results.delta(self._previous_disk_io)
        self._previous_disk_io = total_results
        return current_results
