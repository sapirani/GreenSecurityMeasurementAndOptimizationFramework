import os
import time

from resources_measurement.linux_resources.cgroup_utils import extract_cgroup_relative_path
from resources_measurement.linux_resources.config import SYSTEM_CGROUP_FILE_PATH, FileKeywords
from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader

DEFAULT_NUMBER_OF_CPUS = 1

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

# Sets CPU usage limits for the cgroup. - relevant for cgroup V2
# The format of the file is two values separated by a space: <max> <period>
# <max>: Maximum CPU time (in microseconds) that the cgroup can use in each period.
# <period>: Length of each period in microseconds.
# If <max> is set to max, there is no CPU limit.
# E.g. 50000 100000 - means that we limit the cgroup to 50ms of CPU time every 100ms.
CPU_MAX_FILE_NAME = r"cpu.max"
CPU_MAX_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CPU_MAX_FILE_NAME)

# Specifies the total available run-time within a period for tasks in the cgroup. - relevant for cgroup V1
# A quota sets the maximum amount of CPU time that a cgroup can consume during each period.
# The format of the file is a single integer value in microseconds (-1 indicates no limit).
CPU_CFS_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"
CPU_CFS_QUOTA_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CPU_CFS_QUOTA_FILE_NAME)

# Defines the length of the period for enforcing CPU quotas. - relevant for cgroup V1
# Within each period, the cgroup's CPU usage is limited according to its quota.
# The format of the file is a single integer value in microseconds.
CPU_CFS_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"
CPU_CFS_PERIOD_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CPU_CFS_PERIOD_FILE_NAME)

# The file contains the indices of the cpus that the container can use.
# The file format is a line seperated with commas - each element can be either a single number or a range of cpu indices.
# For example: "0-3,5,7-8"
CPUSET_CPUS_FILE_NAME = r"cpuset.cpus"
CPUSET_CPUS_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CPUSET_CPUS_FILE_NAME)


class LinuxContainerCPUReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__allowed_cpus = self.__get_num_cpus_allowed()
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
        total_possible_ns = time_delta_s * 1e9 * self.__allowed_cpus

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

    def __count_cpus_in_range(self, cpus_string: str) -> int:
        number_of_cpus = 0
        if cpus_string.strip() == "":
            raise Exception("The content of the cpus_string should not be empty")

        for cpus_range in cpus_string.split(','):
            if '-' in cpus_range:  # If it is a range of cpu indices
                start, end = map(int, cpus_range.split('-'))
                number_of_cpus += end - start + 1
            else:  # If it's a single index
                number_of_cpus += 1
        return number_of_cpus

    def __get_num_cpus_allowed(self) -> int:
        try:
            with open(CPUSET_CPUS_FILE_PATH) as f:
                return self.__count_cpus_in_range(f.read().strip())
        except Exception as e:
            print(f"Error when accessing {CPUSET_CPUS_FILE_PATH}: {e}, Using default cpu's count")
            return os.cpu_count() or DEFAULT_NUMBER_OF_CPUS
