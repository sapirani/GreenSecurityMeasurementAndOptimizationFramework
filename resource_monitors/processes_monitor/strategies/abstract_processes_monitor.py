from abc import ABC, abstractmethod
import logging
import time
from dataclasses import dataclass, asdict, astuple
from typing import List, Dict, Tuple, Callable, Iterable

import pandas as pd
import psutil

from utils.general_consts import MB, KB, NUMBER_OF_CORES
from operating_systems.abstract_operating_system import AbstractOSFuncs
from resource_monitors.processes_monitor.process_network_monitor import ProcessNetworkMonitor

logger = logging.getLogger("measurements_logger")


@dataclass
class ProcessMetrics:
    time_since_start: float
    cpu_percent: float = 0
    threads_num: int = 0
    used_memory_mb: float = 0
    used_memory_percent: float = 0
    disk_read_count: int = 0
    disk_write_count: int = 0
    disk_read_kb: float = 0
    disk_write_kb: float = 0
    page_faults: int = 0
    bytes_sent: int = 0
    packets_sent: int = 0
    bytes_received: int = 0
    packets_received: int = 0

    def delta(self, prev_metrics: 'ProcessMetrics') -> 'ProcessMetrics':
        return ProcessMetrics(
            time_since_start=self.time_since_start,
            cpu_percent=self.cpu_percent,
            threads_num=self.threads_num,
            used_memory_mb=self.used_memory_mb,
            used_memory_percent=self.used_memory_percent,
            disk_read_count=self.disk_read_count - prev_metrics.disk_read_count,
            disk_write_count=self.disk_write_count - prev_metrics.disk_write_count,
            disk_read_kb=round(self.disk_read_kb - prev_metrics.disk_read_kb, 3),
            disk_write_kb=round(self.disk_write_kb - prev_metrics.disk_write_kb, 3),
            page_faults=self.page_faults - prev_metrics.page_faults,
            bytes_sent=self.bytes_sent,
            packets_sent=self.packets_sent,
            bytes_received=self.bytes_received,
            packets_received=self.packets_received
        )


class AbstractProcessMonitor(ABC):
    def __init__(
            self,
            process_network_monitor: ProcessNetworkMonitor,
            running_os: AbstractOSFuncs,
            should_ignore_process: Callable[[psutil.Process], bool]
    ):
        self.process_network_monitor = process_network_monitor
        self.running_os = running_os
        self.prev_data_per_process: Dict[Tuple[int, str], ProcessMetrics] = {}
        self.mark_processes = []
        self.should_ignore_process = should_ignore_process
        self.start_time = 0
        self.processes_df = pd.DataFrame()

    def __enter__(self):
        self.process_network_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process_network_monitor.stop()

    def set_processes_to_mark(self, processes: List[psutil.Process]):
        self.mark_processes = processes

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_processes_df(self, processes_df: pd.DataFrame):
        self.processes_df = processes_df

    @property
    def time_since_start(self) -> float:
        return time.time() - self.start_time

    @abstractmethod
    def save_current_processes_statistics(self):
        """
        This function gets all processes running in the system and order them by their cpu usage
        """
        pass

    def monitor_relevant_processes(self, candidate_processes: Iterable[psutil.Process]):
        """
        This function saves the relevant data from the process in dataframe (will be saved later as csv files)
        :param candidate_processes: list of all processes to extract metrics from. They may be filtered by the
            should_ignore_process predicate
        """
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
                    cpu_percent = p.cpu_percent() / NUMBER_OF_CORES
                    process_traffic = self.process_network_monitor.get_network_stats(p)

                    io_stat = p.io_counters()
                    page_faults = self.running_os.get_page_faults(p)

                    if (pid, process_name) not in self.prev_data_per_process:
                        self.prev_data_per_process[(pid, process_name)] = ProcessMetrics(
                            time_since_start=self.time_since_start,
                            disk_read_kb=io_stat.read_bytes / KB,
                            disk_read_count=io_stat.read_count,
                            disk_write_kb=io_stat.write_bytes / KB,
                            disk_write_count=io_stat.write_count,
                            page_faults=page_faults
                        )
                        continue  # remove first sample of process (because cpu_percent is meaningless 0)

                    prev_raw_metrics = self.prev_data_per_process[(pid, process_name)]

                    current_raw_metrics = ProcessMetrics(
                        time_since_start=self.time_since_start,
                        cpu_percent=round(cpu_percent, 2),
                        threads_num=p.num_threads(),
                        used_memory_mb=round(p.memory_info().rss / MB, 3),
                        used_memory_percent=round(p.memory_percent(), 2),
                        disk_read_count=io_stat.read_count,
                        disk_write_count=io_stat.write_count,
                        disk_read_kb=io_stat.read_bytes / KB,
                        disk_write_kb=io_stat.write_bytes / KB,
                        page_faults=page_faults,

                        bytes_sent=(process_traffic.bytes_sent / KB),
                        packets_sent=process_traffic.packets_sent,
                        bytes_received=process_traffic.bytes_received / KB,
                        packets_received=process_traffic.packets_received
                    )

                    current_metrics = current_raw_metrics.delta(prev_raw_metrics)
                    self.processes_df.loc[len(self.processes_df.index)] = [
                        current_metrics.time_since_start,
                        pid,
                        process_name,
                        *astuple(current_metrics)[1:],  # skip time_since_start
                        True if p in self.mark_processes else False
                    ]

                    logger.info(
                        "Process measurements",
                        extra={
                            "pid": pid,
                            "process_name": process_name,
                            "arguments": p.cmdline()[1:],
                            **asdict(current_metrics),
                            **({"process_of_interest": True} if p in self.mark_processes else {})
                        }
                    )

                    self.prev_data_per_process[(pid, process_name)] = current_raw_metrics  # after finishing loop


            # Note, we are just ignoring access denied and other exceptions and do not handle them.
            # There will be no results for those processes
            except (psutil.NoSuchProcess, psutil.AccessDenied, ChildProcessError):
                pass
