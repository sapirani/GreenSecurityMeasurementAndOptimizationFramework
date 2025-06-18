import os
import time

from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader

DEFAULT_NUMBER_OF_CORES = 1

class LinuxContainerCPUReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__last_usage_ns = None
        self.__last_time = None
        self.__number_of_cores = self.__extract_number_of_cores()

    def get_cpu_percent(self) -> float:
        current_usage_ns = self._cgroup_metrics_reader.read_cpu_usage_ns()
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

    def get_number_of_cpu_cores(self) -> int:
        return self.__number_of_cores


    def __parse_cpuset_cpus(self, cpus_string: str) -> int:
        # Example format: "0-3,5,7-8"
        count = 0
        for part in cpus_string.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                count += end - start + 1
            else:
                count += 1
        return count

    def __extract_number_of_cores(self) -> int:
        try:
            with open(self._cgroup_metrics_reader.get_cpu_cores_path()) as f:
                return self.__parse_cpuset_cpus(f.read().strip())
        except Exception:
            return os.cpu_count() or DEFAULT_NUMBER_OF_CORES
