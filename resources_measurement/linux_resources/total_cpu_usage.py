import os
import time

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import extract_cgroup_relative_path
from resources_measurement.linux_resources.config import SYSTEM_CGROUP_FILE_PATH, FileKeywords
from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader


class LinuxContainerCPUReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__last_usage_ns = None
        self.__last_time = None
        self.__cpu_usage_path = self._version.get_cpu_usage_path()

    def get_cpu_percent(self) -> float:
        current_usage_ns = self.__read_cpu_usage_ns()
        current_time = time.time()

        if self.__last_usage_ns is None or self.__last_time is None:
            # First call, initialize tracking variables
            self.__last_usage_ns = current_usage_ns
            self.__last_time = current_time
            return 0.0  # Meaningless value, as per psutil behavior

        # Calculate deltas
        usage_delta_ns = current_usage_ns - self.__last_usage_ns
        time_delta_s = current_time - self.__last_time

        if time_delta_s <= 0:
            return 0.0  # Avoid division by zero or negative time intervals

        # Update tracking variables
        self.__last_usage_ns = current_usage_ns
        self.__last_time = current_time

        # Calculate total possible CPU time in nanoseconds
        total_possible_ns = time_delta_s * 1e9

        # Compute CPU usage percentage
        cpu_percent = (usage_delta_ns / total_possible_ns) * 100
        return cpu_percent

    def __read_cpu_usage_ns(self) -> int:
        return self._version.read_cpu_usage_n(self.__cpu_usage_path)
