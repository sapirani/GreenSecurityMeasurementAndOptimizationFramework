import os
import time

SYS_FS_CGROUP_PATH = r"/sys/fs/cgroup/"

CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CGROUP_CONTROLLERS_FILE_NAME)

CPU_STATS_FILE_NAME = r"cpu.stat"
CPU_STATS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_STATS_FILE_NAME)

CPU_ACCT_USAGE_FILE_NAME = r"cpuacct/cpuacct.usage"
CPU_ACCT_USAGE_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_ACCT_USAGE_FILE_NAME)

CPU_MAX_FILE_NAME = r"cpu.max"
CPU_MAX_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_MAX_FILE_NAME)

CPU_CF_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"
CPU_CF_QUOTA_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_QUOTA_FILE_NAME)

CPU_CF_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"
CPU_CF_PERIOD_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_PERIOD_FILE_NAME)


class FileKeywords:
    V1 = "v1"
    V2 = "v2"
    USAGE_USEC = "usage_usec"
    MAX = "max"


class LinuxContainerCPUReader:
    def __init__(self):
        self.__version = self.__detect_cgroup_version()
        self.__quota_period = self.__read_cpu_limit()
        self.__allowed_cpus = self.__get_num_cpus_allowed()
        self._last_usage_ns = None
        self._last_time = None

    def get_cpu_percent(self):
        current_usage_ns = self.__read_cpu_usage_ns()
        current_time = time.time()

        if self._last_usage_ns is None or self._last_time is None:
            # First call, initialize tracking variables
            self._last_usage_ns = current_usage_ns
            self._last_time = current_time
            return 0.0  # Meaningless value, as per psutil behavior

        # Calculate deltas
        usage_delta_ns = current_usage_ns - self._last_usage_ns
        time_delta_s = current_time - self._last_time

        if time_delta_s <= 0:
            return 0.0  # Avoid division by zero or negative time intervals

        # Update tracking variables
        self._last_usage_ns = current_usage_ns
        self._last_time = current_time

        # Calculate total possible CPU time in nanoseconds
        total_possible_ns = time_delta_s * 1e9 * self.__allowed_cpus

        # Compute CPU usage percentage
        cpu_percent = (usage_delta_ns / total_possible_ns) * 100
        return cpu_percent

    def __read_cpu_usage_ns(self):
        if self.__version == FileKeywords.V2:
            with open(CPU_STATS_FILE_PATH) as f:
                for line in f:
                    if line.startswith(FileKeywords.USAGE_USEC):
                        return int(line.split()[1]) * 1000  # convert to nanoseconds
        else:
            with open(CPU_ACCT_USAGE_FILE_PATH) as f:
                return int(f.read().strip())

    def __detect_cgroup_version(self):
        if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH):
            return FileKeywords.V2
        else:
            return FileKeywords.V1

    def __read_cpu_limit(self):
        if self.__version == FileKeywords.V2:
            try:
                with open(CPU_MAX_FILE_PATH) as f:
                    quota_str, period_str = f.read().strip().split()
                    if quota_str == FileKeywords.MAX:
                        return None  # no limit
                    return int(quota_str), int(period_str)
            except:
                return None
        else:
            try:
                with open(CPU_CF_QUOTA_FILE_PATH) as f:
                    quota = int(f.read().strip())
                with open(CPU_CF_PERIOD_FILE_PATH) as f:
                    period = int(f.read().strip())
                if quota == -1:
                    return None
                return quota, period
            except:
                return None

    def __get_num_cpus_allowed(self):
        if self.__quota_period is None:
            return os.cpu_count()
        quota, period = self.__quota_period
        return quota / period
