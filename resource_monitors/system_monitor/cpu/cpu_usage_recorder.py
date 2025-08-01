import dataclasses
import logging
import threading
from dataclasses import field
from typing import Dict, List
import psutil

from operating_systems.abstract_operating_system import AbstractOSFuncs
from statistics import mean

from resource_monitors import MetricResult, MetricRecorder
from utils.general_consts import LoggerName

logger = logging.getLogger(LoggerName.SYSTEM_METRICS)


@dataclasses.dataclass
class SystemCPUResults(MetricResult):
    mean_cpu_across_cores_percent: float
    sum_cpu_across_cores_percent: float
    number_of_cores: int
    per_core_percent: List[float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Flatten the core percentages"""
        flat = dataclasses.asdict(self)
        # Flatten per_core_percent: core0_percent, core1_percent, ...
        per_core = {
            f"core{idx}_percent": val for idx, val in enumerate(flat.pop("per_core_percent"))
        }
        flat.update(per_core)
        return flat


class SystemCpuUsageRecorder(MetricRecorder):
    def __init__(self, running_os: AbstractOSFuncs, done_scanning_event: threading.Event, is_inside_container: bool):
        self.running_os = running_os
        self.done_scanning_event = done_scanning_event
        self.is_inside_container = is_inside_container

    def get_current_metrics(self) -> SystemCPUResults:
        """
        Saves the total cpu usage of the system
        """
        cpu_per_core: List[float] = psutil.cpu_percent(percpu=True)     # noqa
        if self.is_inside_container:
            try:    # in case of measuring cpu in a windows container
                cpu_sum_across_cores = self.running_os.get_container_total_cpu_usage()
                # TODO: PUT NUMBER OF CORES IN THE CONSTRUCTOR
                number_of_cores = self.running_os.get_container_number_of_cores()
                cpu_mean_across_cores = cpu_sum_across_cores / number_of_cores
            except NotImplementedError as e:
                print(f"Error occurred: {str(e)}")
                self.done_scanning_event.set()
        else:
            cpu_mean_across_cores = mean(cpu_per_core)
            cpu_sum_across_cores = sum(cpu_per_core)
            number_of_cores = len(cpu_per_core)

        cpu_sum_across_cores = round(cpu_sum_across_cores, 2)
        cpu_mean_across_cores = round(cpu_mean_across_cores, 2)

        return SystemCPUResults(
            mean_cpu_across_cores_percent=cpu_mean_across_cores,
            sum_cpu_across_cores_percent=cpu_sum_across_cores,
            number_of_cores=number_of_cores,
            per_core_percent=cpu_per_core
        )
