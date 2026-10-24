import psutil

from DTOs.scanner_counters.system.disk_results import SystemDiskResults
from resource_usage_recorder.container_recorder.linux_resources.disk_usage import LinuxContainerDiskReader
from utils.general_consts import KB
from resource_usage_recorder import MetricRecorder

# TODO: CREATE 2 READERS - ONE FOR THE SYSTEM-WIDE AND ONE FOR THE CONTAINER AND INJECT THEM HERE
#   THERE WILL BE A FACTORY FUNCTION OUTSIDE OF THIS CLASS THAT CHECKS IF INSIDE CONTINER AND RETURNS THE
#   APPROPRIATE RECORDER
class SystemDiskUsageRecorder(MetricRecorder):
    def __init__(self, is_inside_container: bool):
        super().__init__()
        self._is_inside_container = is_inside_container
        self._previous_disk_io = self._get_total_results()

        self._container_disk_reader = None
        if self._is_inside_container:
            self._container_disk_reader = LinuxContainerDiskReader()

    def _get_total_results(self) -> SystemDiskResults:
        if self._is_inside_container:
            return self._container_disk_reader.get_total_disk_counters()
        else:
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
