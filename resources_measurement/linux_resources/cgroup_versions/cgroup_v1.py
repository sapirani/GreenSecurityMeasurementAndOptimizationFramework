import os

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupMetricReader, \
    CGROUP_TYPE_PATH
from resources_measurement.linux_resources.cgroup_versions.cgroup_entry import CgroupEntry
from resources_measurement.linux_resources.cgroup_versions.common_paths import CPUSET_CPUS_FILE_NAME, \
    SYSTEM_CGROUP_DIR_PATH

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
    __CPU_ACCT_USAGE_FILE_NAME_V1 = r"cpuacct.usage"

    __CPUSET_CGROUP_SUBSYSTEM = "cpuset"

    def get_version(self) -> str:
        return CGROUP_V1_NAME

    def _is_cgroup_dir(self, cgroup_entry: CgroupEntry) -> bool:
        return cgroup_entry.subsystems == self.__CGROUP_V1_MEMORY_CONTROLLERS or \
                cgroup_entry.subsystems == self.__CGROUP_V1_CPU_CONTROLLERS

    def read_cpu_usage_ns(self) -> int:
        try:
            with open(self._cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {self._cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def _get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_ACCT_USAGE_FILE_NAME_V1)

    def get_cpu_cores_path(self) -> str:
        cpuset_path = self.__get_cpuset_cgroup_path()
        return os.path.join(SYSTEM_CGROUP_DIR_PATH, self.__CPUSET_CGROUP_SUBSYSTEM, cpuset_path, CPUSET_CPUS_FILE_NAME)

    def __get_cpuset_cgroup_path(self) -> str:
        for entry in self._get_all_cgroup_entries():
            if self.__CPUSET_CGROUP_SUBSYSTEM in entry.subsystems.split(','):
                return entry.cgroup_path.lstrip('/')
        return ""

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V1)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V1)

    def is_container_memory_limited(self, limit: str) -> bool:
        limit_number = int(limit)
        return limit_number >= 2 ** 60
