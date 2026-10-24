from DTOs.scanner_counters.system.disk_results import SystemDiskResults
from resource_usage_recorder.container_recorder.linux_resources.abstract_resource_usage import \
    AbstractLinuxContainerResourceReader



class LinuxContainerDiskReader(AbstractLinuxContainerResourceReader):
    def get_total_disk_counters(self) -> SystemDiskResults:
        return self._cgroup_metrics_reader.get_total_disk_counters()
