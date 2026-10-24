from dataclasses import dataclass

from DTOs.scanner_counters.metric_results import MetricResult


@dataclass
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
