import time

from resource_usage_recorder.container_recorder.linux_resources.abstract_resource_usage import \
    AbstractLinuxContainerResourceReader

DEFAULT_NUMBER_OF_CORES = 1


class LinuxContainerCPUReader(AbstractLinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__last_usage_ns = None
        self.__last_time = None
        self.__number_of_cores = self._cgroup_metrics_reader.get_container_vcores()

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

    def get_number_of_cpu_cores(self) -> float:
        return self.__number_of_cores
