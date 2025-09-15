from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Iterable
import psutil

from resource_usage_recorder import MetricResult
from resource_usage_recorder.processes_recorder.process_network_usage_recorder import ProcessNetworkUsageRecorder
from utils.general_consts import MB, KB, NUMBER_OF_CORES, LoggerName
from operating_systems.abstract_operating_system import AbstractOSFuncs

logger = logging.getLogger(LoggerName.PROCESS_METRICS)


@dataclass
class ProcessMetrics(MetricResult):
    pid: int
    process_name: str
    arguments: List[str]
    cpu_percent_sum_across_cores: float = 0     # can exceed 100% in case where process utilizes more than one core
    cpu_percent_mean_across_cores: float = 0
    threads_num: int = 0
    used_memory_mb: float = 0
    used_memory_percent: float = 0
    disk_read_count: int = 0
    disk_write_count: int = 0
    disk_read_kb: float = 0
    disk_write_kb: float = 0
    page_faults: int = 0
    network_kb_sent: float = 0
    packets_sent: int = 0
    network_kb_received: float = 0
    packets_received: int = 0
    process_of_interest: bool = False

    def delta(self, prev_metrics: 'ProcessMetrics') -> 'ProcessMetrics':
        return ProcessMetrics(
            pid=self.pid,
            process_name=self.process_name,
            arguments=self.arguments,
            cpu_percent_sum_across_cores=self.cpu_percent_sum_across_cores,
            cpu_percent_mean_across_cores=self.cpu_percent_mean_across_cores,
            threads_num=self.threads_num,
            used_memory_mb=self.used_memory_mb,
            used_memory_percent=self.used_memory_percent,
            disk_read_count=self.disk_read_count - prev_metrics.disk_read_count,
            disk_write_count=self.disk_write_count - prev_metrics.disk_write_count,
            disk_read_kb=round(self.disk_read_kb - prev_metrics.disk_read_kb, 3),
            disk_write_kb=round(self.disk_write_kb - prev_metrics.disk_write_kb, 3),
            page_faults=self.page_faults - prev_metrics.page_faults,
            network_kb_sent=self.network_kb_sent,
            packets_sent=self.packets_sent,
            network_kb_received=self.network_kb_received,
            packets_received=self.packets_received,
            process_of_interest=self.process_of_interest
        )


class AbstractProcessResourceUsageRecorder(ABC):
    def __init__(
            self,
            process_network_usage_recorder: ProcessNetworkUsageRecorder,
            running_os: AbstractOSFuncs,
            should_ignore_process: Callable[[psutil.Process], bool]
    ):
        self.process_network_usage_recorder = process_network_usage_recorder
        self.running_os = running_os
        self.prev_data_per_process: Dict[Tuple[int, str], ProcessMetrics] = {}
        self.mark_processes = []
        self.should_ignore_process = should_ignore_process

    def __enter__(self):
        self.process_network_usage_recorder.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process_network_usage_recorder.stop()

    def set_processes_to_mark(self, processes: List[psutil.Process]):
        self.mark_processes = processes

    @abstractmethod
    def get_current_metrics(self) -> List[ProcessMetrics]:
        """
        This function gets telemetry from all desired processes.
        """
        pass

    def _get_current_metrics(self, candidate_processes: Iterable[psutil.Process]) -> List[ProcessMetrics]:
        """
        This function Returns telemetry about processes.
        :param candidate_processes: list of all processes to extract metrics from. They may be filtered by the
            should_ignore_process predicate
        """
        processes_results = []

        for p in candidate_processes:
            try:
                if self.should_ignore_process(p) and not (p in self.mark_processes):
                    continue
            except Exception:
                continue

            # While fetching the processes, some subprocesses may exit
            # Hence we need to put this code in try-except block
            try:
                # oneshot to improve info retrieve efficiency
                with p.oneshot():
                    pid = p.pid
                    process_name = p.name()
                    process_args = p.cmdline()[1:]
                    process_args = None
                    process_of_interest = True if p in self.mark_processes else False
                    cpu_percent_sum_across_cores = round(p.cpu_percent(), 2)
                    process_traffic = self.process_network_usage_recorder.get_current_network_stats(p)

                    io_stat = p.io_counters()
                    page_faults = self.running_os.get_page_faults(p)

                    if (pid, process_name) not in self.prev_data_per_process:
                        self.prev_data_per_process[(pid, process_name)] = ProcessMetrics(
                            pid=pid,
                            process_name=process_name,
                            arguments=process_args,
                            disk_read_kb=io_stat.read_bytes / KB,
                            disk_read_count=io_stat.read_count,
                            disk_write_kb=io_stat.write_bytes / KB,
                            disk_write_count=io_stat.write_count,
                            page_faults=page_faults
                        )
                        continue  # remove first sample of process (because cpu_percent is meaningless 0)

                    prev_raw_metrics = self.prev_data_per_process[(pid, process_name)]

                    current_raw_metrics = ProcessMetrics(
                        pid=pid,
                        process_name=process_name,
                        arguments=process_args,
                        cpu_percent_sum_across_cores=cpu_percent_sum_across_cores,
                        cpu_percent_mean_across_cores=cpu_percent_sum_across_cores / NUMBER_OF_CORES,
                        threads_num=p.num_threads(),
                        used_memory_mb=round(p.memory_info().rss / MB, 3),
                        # TODO: maybe should use uss/pss instead rss?
                        used_memory_percent=round(p.memory_percent(), 2),
                        disk_read_count=io_stat.read_count,
                        disk_write_count=io_stat.write_count,
                        disk_read_kb=io_stat.read_bytes / KB,
                        disk_write_kb=io_stat.write_bytes / KB,
                        page_faults=page_faults,
                        network_kb_sent=process_traffic.bytes_sent / KB,
                        packets_sent=process_traffic.packets_sent,
                        network_kb_received=process_traffic.bytes_received / KB,
                        packets_received=process_traffic.packets_received,
                        process_of_interest=process_of_interest
                    )

                    current_metrics = current_raw_metrics.delta(prev_raw_metrics)
                    processes_results.append(current_metrics)
                    self.prev_data_per_process[(pid, process_name)] = current_raw_metrics  # after finishing loop

            # Note, we are just ignoring access denied and other exceptions and do not handle them.
            # There will be no results for those processes
            except (psutil.NoSuchProcess, psutil.AccessDenied, ChildProcessError):
                pass

        return processes_results
