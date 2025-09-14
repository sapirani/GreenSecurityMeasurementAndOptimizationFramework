import dataclasses
import logging
import threading
from dataclasses import field
from typing import Dict, List
import psutil

from operating_systems.abstract_operating_system import AbstractOSFuncs
from statistics import mean

from resource_usage_recorder import MetricResult, MetricRecorder
from utils.general_consts import LoggerName
from dataclasses import dataclass


logger = logging.getLogger(LoggerName.SYSTEM_METRICS)


@dataclass
class SystemCPUResults(MetricResult):
    cpu_percent_mean_across_cores: float
    cpu_percent_sum_across_cores: float
    number_of_cores: int
    per_core_percent: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Flatten the core percentages"""
        flat = dataclasses.asdict(self)
        # Flatten per_core_percent: core0_percent, core1_percent, ...
        per_core = {
            # CHANGE get_core_name in general consts if you change this variable name
            f"core_{idx}_percent": val for idx, val in enumerate(flat.pop("per_core_percent"))
        }
        flat.update(per_core)
        return flat


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
