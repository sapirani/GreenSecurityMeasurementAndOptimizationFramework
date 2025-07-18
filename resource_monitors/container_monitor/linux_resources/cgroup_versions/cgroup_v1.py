import os

from resource_monitors.container_monitor.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupMetricReader

CGROUP_V1_NAME = "V1"


class CgroupMetricReaderV1(CgroupMetricReader):
    __CGROUP_V1_CPU_CONTROLLERS = "cpu,cpuacct"
    __CGROUP_V1_MEMORY_CONTROLLERS = "memory"

    # Reports the current memory usage in the cgroup, in bytes.
    # Includes memory used by all descendant cgroups.
    # The format of this file is a single integer value in bytes.
    __MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"

    # Shows the memory limit set for the cgroup in bytes.
    # If set to a very large number (e.g., 9223372036854771712), it means no limit.
    # The format of this file is a single integer value in bytes.
    __MEMORY_MAX_FILE_NAME_V1 = "memory/memory.limit_in_bytes"

    # Reports the total CPU time consumed by tasks in the cgroup.
    # "acct" stands for "accounting" and refers to the mechanism for tracking resource usage.
    # Therefore, cpuacct.usage provides a report on the total CPU time used by all processes managed by the cgroup.
    # The format of the file is a single integer value representing nanoseconds.
    __CPU_ACCT_USAGE_FILE_NAME_V1 = r"cpuacct/cpuacct.usage"

    # Specifies the total available run-time within a period for tasks in the cgroup. - relevant for cgroup V1
    # A quota sets the maximum amount of CPU time that a cgroup can consume during each period.
    # The format of the file is a single integer value in microseconds (-1 indicates no limit).
    __CPU_CF_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"

    # Defines the length of the period for enforcing CPU quotas. - relevant for cgroup V1
    # Within each period, the cgroup's CPU usage is limited according to its quota.
    # The format of the file is a single integer value in microseconds.
    __CPU_CF_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"

    __NO_QUOTA_LIMIT = -1

    def get_version(self) -> str:
        return CGROUP_V1_NAME

    def read_cpu_usage_ns(self) -> int:
        try:
            with open(self._cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {self._cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def _get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_ACCT_USAGE_FILE_NAME_V1)

    def get_container_vcores(self) -> float:
        try:
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_CF_QUOTA_FILE_NAME)) as f:
                quota = int(f.read().strip())
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_CF_PERIOD_FILE_NAME)) as f:
                period = int(f.read().strip())
            if quota == self.__NO_QUOTA_LIMIT:
                return os.cpu_count()
            return quota / period
        except Exception as e:
            print(f"Warning: Cannot get number of containers due to error: {e}, Using cpu_count()")
            return os.cpu_count()

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V1)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V1)

    def is_container_memory_limited(self, limit: str) -> bool:
        limit_number = int(limit)
        return limit_number >= 2 ** 60
