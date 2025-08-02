from dataclasses import dataclass


import psutil

from operating_systems.abstract_operating_system import AbstractOSFuncs
from utils.general_consts import GB
from resource_usage_recorder import MetricResult, MetricRecorder


@dataclass
class SystemMemoryResults(MetricResult):
    total_memory_gb: float
    total_memory_percent: float


class SystemMemoryUsageRecorder(MetricRecorder):
    def __init__(self, running_os: AbstractOSFuncs, is_inside_container: bool):
        self.running_os = running_os
        self.is_inside_container = is_inside_container

    def get_current_metrics(self) -> SystemMemoryResults:
        if self.is_inside_container:
            memory_used_bytes, memory_used_percent = self.running_os.get_container_total_memory_usage()
        else:
            vm = psutil.virtual_memory()
            memory_used_bytes, memory_used_percent = vm.used, vm.percent

        return SystemMemoryResults(
            total_memory_gb=round(memory_used_bytes / GB, 3),
            total_memory_percent=memory_used_percent
        )
