import os
import time
from typing import Optional

DEFAULT_NUMBER_OF_CPUS = 1

# A cgroup is a feature that allows you to allocate, limit, and monitor system resources among user-defined groups of processes.
# Enables control over resource distribution, ensuring that no single group can monopolize the system resources.
SYS_FS_CGROUP_PATH = r"/sys/fs/cgroup/"

# Lists the available controllers (e.g., cpu, memory) that can be enabled in the current cgroup.
# The format of the file is a single line with space-separated controller names.
# E.g. cpu io memory
# Presence of this file indicates that the system is using cgroup v2.
CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CGROUP_CONTROLLERS_FILE_NAME)

# Provides CPU usage statistics for the cgroup.
# The format of the file is key-value pairs, one per line.
# E.g.
#   usage_usec 123456789
#   user_usec 12345678
#   system_usec 11111111
# The interesting key is usage_usec, its value is the total CPU time consumed by all tasks in the cgroup, in microseconds.
CPU_STATS_FILE_NAME = r"cpu.stat"
CPU_STATS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_STATS_FILE_NAME)

# Reports the total CPU time consumed by tasks in the cgroup. - relevant for cgroup V1
# The format of the file is a single integer value representing nanoseconds.
CPU_ACCT_USAGE_FILE_NAME = r"cpuacct.usage"

#CPU_ACCT_USAGE_FILE_NAME = r"cpuacct/cpuacct.usage"
CPU_ACCT_USAGE_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_ACCT_USAGE_FILE_NAME)

# Sets CPU usage limits for the cgroup. - relevant for cgroup V2
# The format of the file is two values separated by a space: <max> <period>
# <max>: Maximum CPU time (in microseconds) that the cgroup can use in each period.
# <period>: Length of each period in microseconds.
# If <max> is set to max, there is no CPU limit.
# E.g. 50000 100000 - means that we limit the cgroup to 50ms of CPU time every 100ms.
CPU_MAX_FILE_NAME = r"cpu.max"
CPU_MAX_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_MAX_FILE_NAME)

# Specifies the total available run-time within a period for tasks in the cgroup. - relevant for cgroup V1
# A quota sets the maximum amount of CPU time that a cgroup can consume during each period.
# The format of the file is a single integer value in microseconds (-1 indicates no limit).
CPU_CF_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"
CPU_CF_QUOTA_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_QUOTA_FILE_NAME)

# Defines the length of the period for enforcing CPU quotas. - relevant for cgroup V1
# Within each period, the cgroup's CPU usage is limited according to its quota.
# The format of the file is a single integer value in microseconds.
CPU_CF_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"
CPU_CF_PERIOD_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_PERIOD_FILE_NAME)


# Contains details on the cgroup of the container.
# The file format is the single line:
# hierarchy-ID : controllers : cgroup-path -> WHERE:
# Hierarchy ID can be 0 (for cgroup v2), or 2, 3, etc. in v1
# Controller(s)	can be cpu,cpuacct or empty string ("") for v2
# Path to cgroup can be /docker/<container-id> or /
CGROUP_IN_CONTAINER_PATH = r"/proc/self/cgroup"
SYSTEM_CGROUP_FILE_PATH = r"/sys/fs/cgroup/"

NEEDED_OPERATIONS = "cpu,cpuacct"


class FileKeywords:
    V1 = "v1"
    V2 = "v2"
    USAGE_USEC = "usage_usec"
    MAX = "max"


class LinuxContainerCPUReader:
    def __init__(self):
        self.__version = self.__detect_cgroup_version()
        self._cgroup_path = self.__get_actual_cgroup_cpu_path(self.__version)
        self._cpu_stat_path = os.path.join(self._cgroup_path, CPU_STATS_FILE_NAME)
        self._cpuacct_usage_path = os.path.join(self._cgroup_path, CPU_ACCT_USAGE_FILE_NAME)

        # self.__quota_period = self.__read_cpu_limit()
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

    def __read_cpu_usage_ns(self) -> Optional[int]:
        if self.__version == FileKeywords.V2:
            with open(self._cpu_stat_path) as f:
                for line in f:
                    if line.startswith(FileKeywords.USAGE_USEC):
                        return int(line.split()[1]) * 1000  # convert to nanoseconds
        else:
            with open(self._cpuacct_usage_path) as f:
                return int(f.read().strip())

    def __detect_cgroup_version(self) -> str:
        if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH):
            return FileKeywords.V2
        else:
            return FileKeywords.V1

    def __get_actual_cgroup_cpu_path(self, version: str) -> str:
        with open(CGROUP_IN_CONTAINER_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if version == FileKeywords.V2:
                    if len(parts) == 3 and parts[0] == "0":
                        return os.path.join(SYSTEM_CGROUP_FILE_PATH, parts[2].lstrip("/"))
                else:
                    if len(parts) == 3 and parts[1] == NEEDED_OPERATIONS:
                        return os.path.join(SYSTEM_CGROUP_FILE_PATH, parts[2].lstrip("/"))
        return SYSTEM_CGROUP_FILE_PATH  # fallback (might be incorrect)


    # def __read_cpu_limit(self) -> Optional[tuple[int, int]]:
    #     if self.__version == FileKeywords.V2:
    #         try:
    #             with open(CPU_MAX_FILE_PATH) as f:
    #                 quota_str, period_str = f.read().strip().split()
    #                 if quota_str == FileKeywords.MAX:
    #                     return None  # no limit
    #                 return int(quota_str), int(period_str)
    #         except:
    #             return None
    #     else:
    #         try:
    #             with open(CPU_CF_QUOTA_FILE_PATH) as f:
    #                 quota = int(f.read().strip())
    #             with open(CPU_CF_PERIOD_FILE_PATH) as f:
    #                 period = int(f.read().strip())
    #             if quota == -1:
    #                 return None
    #             return quota, period
    #         except:
    #             return None

    def __get_num_cpus_allowed(self) -> int:
        cpu_count = os.cpu_count()
        return cpu_count if cpu_count is not None else DEFAULT_NUMBER_OF_CPUS
        # if self.__quota_period is None:
        #     cpu_count = os.cpu_count()
        #     return cpu_count if cpu_count is not None else DEFAULT_NUMBER_OF_CPUS
        # quota, period = self.__quota_period
        # return int(quota / period)
