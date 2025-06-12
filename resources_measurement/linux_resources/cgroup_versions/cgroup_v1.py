import os

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion
from resources_measurement.linux_resources.cgroup_versions.cgroup_entry import CgroupEntry

CGROUP_V1_CPU_CONTROLLERS = "cpu,cpuacct"
CGROUP_V1_MEMORY_CONTROLLERS = "memory"

CGROUP_V1_NAME = "V1"


class CgroupV1(CgroupVersion):
    __MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"
    __MEMORY_MAX_FILE_NAME_V1 = "memory/memory.limit_in_bytes"

    # Reports the total CPU time consumed by tasks in the cgroup.
    # "acct" stands for "accounting" and refers to the mechanism for tracking resource usage.
    # Therefore, cpuacct.usage provides a report on the total CPU time used by all processes managed by the cgroup.
    # The format of the file is a single integer value representing nanoseconds.
    __CPU_ACCT_USAGE_FILE_NAME_V1 = r"cpuacct.usage"

    def get_version(self) -> str:
        return CGROUP_V1_NAME

    def _is_cgroup_dir(self, cgroup_entry: CgroupEntry) -> bool:
        return (cgroup_entry.subsystems == CGROUP_V1_MEMORY_CONTROLLERS) or (cgroup_entry.subsystems == CGROUP_V1_CPU_CONTROLLERS)

    def read_cpu_usage_ns(self, cpu_usage_file_path: str) -> int:
        try:
            with open(cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_ACCT_USAGE_FILE_NAME_V1)

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V1)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V1)
