import os

from resource_usage_recorder.container_recorder.linux_resources.cgroup_versions.abstract_cgroup_version import \
    CgroupMetricReader

CGROUP_V2_NAME = "V2"



class CgroupMetricReaderV2(CgroupMetricReader):
    __USAGE_USEC_V2 = "usage_usec"
    __NO_MEMORY_LIMIT = "max"

    __CGROUP_V2_IDENTIFIER = "0"

    # Reports the current memory usage of the cgroup in bytes,
    # including memory used by all processes in the cgroup and its descendants.
    # The file format is a single integer representing bytes of memory currently used.
    __MEMORY_USAGE_FILE_NAME_V2 = "memory.current"

    # Sets Memory usage limits for the cgroup.
    # The format of the file is either a single integer (in bytes) or the string max.
    # <max>: Maximum memory bytes that the cgroup can use.
    # If <max> is set to max, there is no memory limit.
    __MEMORY_MAX_FILE_NAME_V2 = "memory.max"

    # Provides CPU usage statistics for the cgroup.
    # The format of the file is key-value pairs, one per line.
    # E.g.
    #   usage_usec 123456789
    #   user_usec 12345678
    #   system_usec 11111111
    # The interesting key is usage_usec, its value is the total CPU time consumed by all tasks in the cgroup, in microseconds.
    __CPU_STATS_FILE_NAME_V2 = "cpu.stat"

    # Sets CPU usage limits for the cgroup. - relevant for cgroup V2
    # The format of the file is two values separated by a space: <max> <period>
    # <max>: Maximum CPU time (in microseconds) that the cgroup can use in each period.
    # <period>: Length of each period in microseconds.
    # If <max> is set to max, there is no CPU limit.
    __CPU_MAX_FILE_NAME = r"cpu.max"

    __NO_QUOTA_LIMIT = "max"

    def get_version(self) -> str:
        return CGROUP_V2_NAME

    def read_cpu_usage_ns(self) -> int:
        try:
            with open(self._cpu_usage_file_path) as f:
                for line in f:
                    if line.startswith(self.__USAGE_USEC_V2):
                        return int(line.split()[1]) * 1000  # convert to nanoseconds 143512538

            return 0
        except Exception as e:
            raise ValueError(f"The file {self._cpu_usage_file_path} does not exist or is not readable in Cgroup V2.")

    def _get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_STATS_FILE_NAME_V2)

    def get_container_vcores(self) -> float:
        try:
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_MAX_FILE_NAME)) as f:
                quota_str, period_str = f.read().strip().split()
                if quota_str == self.__NO_QUOTA_LIMIT:
                    return os.cpu_count()  # no limit
                return float(quota_str) / float(period_str)
        except Exception as e:
            print(f"Warning: Cannot get number of containers due to error: {e}, Using cpu_count()")
            return os.cpu_count()

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V2)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V2)

    def is_container_memory_limited(self, limit: str) -> bool:
        return limit == self.__NO_MEMORY_LIMIT
