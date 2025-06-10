import os
import time

from resources_measurement.linux_resources.cgroup_utils import extract_cgroup_relative_path
from resources_measurement.linux_resources.config import SYSTEM_CGROUP_FILE_PATH, FileKeywords
from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader

# Provides CPU usage statistics for the cgroup.
# The format of the file is key-value pairs, one per line.
# E.g.
#   usage_usec 123456789
#   user_usec 12345678
#   system_usec 11111111
# The interesting key is usage_usec, its value is the total CPU time consumed by all tasks in the cgroup, in microseconds.
CPU_STATS_FILE_NAME = r"cpu.stat"

# Reports the total CPU time consumed by tasks in the cgroup. - relevant for cgroup V1
# The format of the file is a single integer value representing nanoseconds.
CPU_ACCT_USAGE_FILE_NAME = r"cpuacct.usage"


class LinuxContainerCPUReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__last_usage_ns = None
        self.__last_time = None

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
        current_cpu_usage = 0
        try:
            if self._version == FileKeywords.V2:
                with open(self._resource_usage_path) as f:
                    for line in f:
                        if line.startswith(FileKeywords.USAGE_USEC):
                            current_cpu_usage = int(line.split()[1]) * 1000  # convert to nanoseconds 143512538
                            break
            else:
                with open(self._resource_usage_path) as f:
                    current_cpu_usage = int(f.read().strip())
            return current_cpu_usage
        except Exception as e:
            print(f"Error when accessing {self._resource_usage_path}: {e}")
            return 0

    def _get_resource_file_path(self, version: str) -> str:
        path_to_cgroup_dir = extract_cgroup_relative_path(version, FileKeywords.CGROUP_V1_CPU_IDENTIFIER)
        if version == FileKeywords.V2:
            return os.path.join(path_to_cgroup_dir, CPU_STATS_FILE_NAME)
        else:
            return os.path.join(path_to_cgroup_dir, CPU_ACCT_USAGE_FILE_NAME)
