import logging
import threading
from typing import List
import psutil

from DTOs.scanner_counters.system.cpu_results import SystemCPUResults
from operating_systems.abstract_operating_system import AbstractOSFuncs
from statistics import mean

from resource_usage_recorder import MetricRecorder
from DTOs.logging.consts import LoggerName

logger = logging.getLogger(LoggerName.SYSTEM_METRICS)


class SystemCpuUsageRecorder(MetricRecorder):
    def __init__(self, running_os: AbstractOSFuncs, done_scanning_event: threading.Event, is_inside_container: bool):
        self.running_os = running_os
        self.done_scanning_event = done_scanning_event
        self.is_inside_container = is_inside_container
        if is_inside_container:
            self.number_of_cores = running_os.get_container_number_of_cores()
        else:
            self.number_of_cores = psutil.cpu_count(logical=True)

    def get_current_metrics(self) -> SystemCPUResults:
        """
        Saves the total cpu usage of the system
        """
        cpu_per_core: List[float] = psutil.cpu_percent(percpu=True)     # noqa
        cpu_sum_across_cores = -1
        cpu_mean_across_cores = -1
        if self.is_inside_container:
            try:    # in case of measuring cpu in a windows container
                cpu_sum_across_cores = self.running_os.get_container_total_cpu_usage()
                cpu_mean_across_cores = cpu_sum_across_cores / self.number_of_cores
            except NotImplementedError as e:
                print(f"Error occurred: {str(e)}")
                self.done_scanning_event.set()
        else:
            cpu_mean_across_cores = mean(cpu_per_core)
            cpu_sum_across_cores = sum(cpu_per_core)

        cpu_percent_sum_across_cores = round(cpu_sum_across_cores, 2)
        cpu_percent_mean_across_cores = round(cpu_mean_across_cores, 2)

        return SystemCPUResults(
            cpu_percent_mean_across_cores=cpu_percent_mean_across_cores,
            cpu_percent_sum_across_cores=cpu_percent_sum_across_cores,
            number_of_cores=self.number_of_cores,
            per_core_percent=cpu_per_core
        )
