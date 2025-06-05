import os
import time


DEFAULT_NUMBER_OF_CPUS = 1

# A cgroup is a feature that allows you to allocate, limit, and monitor system resources among user-defined groups of processes.
# Enables control over resource distribution, ensuring that no single group can monopolize the system resources.
SYSTEM_CGROUP_FILE_PATH = r"/sys/fs/cgroup/"

# Lists the available controllers (e.g., cpu, memory) that can be enabled in the current cgroup.
# The format of the file is a single line with space-separated controller names.
# E.g. cpu io memory
# Presence of this file indicates that the system is using cgroup v2.
CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CGROUP_CONTROLLERS_FILE_NAME)

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

# Contains details on the cgroup of the container.
# The file format is the single line:
# hierarchy-ID : controllers : cgroup-path -> WHERE:
# Hierarchy ID can be 0 (for cgroup v2), or 2, 3, etc. in v1
# Controller(s)	can be cpu,cpuacct or empty string ("") for v2
# Path to cgroup can be /docker/<container-id> or /
CGROUP_IN_CONTAINER_PATH = r"/proc/self/cgroup"

# The file contains the indices of the cpus that the container can use.
# The file format is a line seperated with commas - each element can be either a single number or a range of cpu indices.
# For example: "0-3,5,7-8"
CPUSET_CPUS_FILE_NAME = r"cpuset.cpus"
CPUSET_CPUS_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CPUSET_CPUS_FILE_NAME)


class ProcCgroupFileConsts:
    NUMBER_OF_ELEMENTS = 3
    HIERARCHY_INDEX = 0
    CONTROLLERS_INDEX = 1
    CGROUP_PATH_INDEX = 2


class FileKeywords:
    V1 = "v1"
    V2 = "v2"
    USAGE_USEC = "usage_usec"
    MAX = "max"

    CGROUP_V1_IDENTIFIER = "cpu,cpuacct"
    CGROUP_V2_IDENTIFIER = "0"


class LinuxContainerCPUReader:
    def __init__(self):
        self.__version = self.__detect_cgroup_version()
        self.__cpu_stats_path = self.__get_cpu_file_path(self.__version)
        self.__cpu_limit = self.__get_cpu_limit()
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
        total_possible_ns = time_delta_s * 1e9 * self.__cpu_limit

        # Compute CPU usage percentage
        cpu_percent = (usage_delta_ns / total_possible_ns) * 100
        return cpu_percent

    def __read_cpu_usage_ns(self) -> int:
        current_cpu_usage = 0
        try:
            if self.__version == FileKeywords.V2:
                with open(self.__cpu_stats_path) as f:
                    for line in f:
                        if line.startswith(FileKeywords.USAGE_USEC):
                            current_cpu_usage = int(line.split()[1]) * 1000  # convert to nanoseconds 143512538
                            break
            else:
                with open(self.__cpu_stats_path) as f:
                    current_cpu_usage = int(f.read().strip())
            return current_cpu_usage
        except Exception as e:
            print(f"Error when accessing {self.__cpu_stats_path}: {e}")
            return 0

    def __detect_cgroup_version(self) -> str:
        return FileKeywords.V2 if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH) else FileKeywords.V1

    def __extract_cgroup_cpu_relative_path(self, version: str) -> str:
        with open(CGROUP_IN_CONTAINER_PATH, "r") as f:
            for line in f:
                proc_cgroup_parts = line.strip().split(":")
                if len(proc_cgroup_parts) != ProcCgroupFileConsts.NUMBER_OF_ELEMENTS:
                    continue
                else:
                    hierarchy = proc_cgroup_parts[ProcCgroupFileConsts.HIERARCHY_INDEX]
                    controllers = proc_cgroup_parts[ProcCgroupFileConsts.CONTROLLERS_INDEX]
                    cgroup_path = proc_cgroup_parts[ProcCgroupFileConsts.CGROUP_PATH_INDEX].lstrip("/")

                    if (version == FileKeywords.V2 and hierarchy == FileKeywords.CGROUP_V2_IDENTIFIER) or \
                            (version == FileKeywords.V1 and controllers == FileKeywords.CGROUP_V1_IDENTIFIER):
                        return os.path.join(SYSTEM_CGROUP_FILE_PATH, cgroup_path)

        return SYSTEM_CGROUP_FILE_PATH

    def __get_cpu_file_path(self, version: str) -> str:
        path_to_cgroup_dir = self.__extract_cgroup_cpu_relative_path(version)
        if version == FileKeywords.V2:
            return os.path.join(path_to_cgroup_dir, CPU_STATS_FILE_NAME)
        else:
            return os.path.join(path_to_cgroup_dir, CPU_ACCT_USAGE_FILE_NAME)

    def __get_cpu_limit(self) -> float:
        cpu_quota, cpu_period = self.__read_cpu_quota_and_period()
        if cpu_quota is not None and cpu_period is not None:
            cpu_limit = cpu_quota / cpu_period
        else:
            cpu_limit = self.__get_num_cpus_allowed()
        return cpu_limit

    def __read_cpu_quota_and_period(self):
        try:
            if self.__version == FileKeywords.V2:
                with open(CPU_MAX_FILE_PATH) as f:
                    quota_str, period_str = f.read().strip().split()
                    if quota_str == FileKeywords.MAX:
                        return None, None
                    return int(quota_str), int(period_str)
            else:
                with open(CPU_CFS_QUOTA_FILE_PATH) as f:
                    quota = int(f.read().strip())
                with open(CPU_CFS_PERIOD_FILE_PATH) as f:
                    period = int(f.read().strip())
                if quota == -1:
                    return None, None
                return quota, period
        except Exception as e:
            print(f"Error reading CPU quota/period: {e}")
            return None, None

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
