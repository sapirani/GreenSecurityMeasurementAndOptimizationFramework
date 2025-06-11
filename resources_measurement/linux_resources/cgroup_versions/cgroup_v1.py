import os

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion


class FilePathsV1:
    MEMORY_USAGE_FILE_NAME = "memory/memory.usage_in_bytes"
    MEMORY_MAX_FILE_NAME = "memory/memory.limit_in_bytes"

    # Reports the total CPU time consumed by tasks in the cgroup.
    # "acct" stands for "accounting" and refers to the mechanism for tracking resource usage.
    # Therefore, cpuacct.usage provides a report on the total CPU time used by all processes managed by the cgroup.
    # The format of the file is a single integer value representing nanoseconds.
    CPU_ACCT_USAGE_FILE_NAME = r"cpuacct.usage"


CGROUP_V1_NAME = "V1"


class CgroupV1(CgroupVersion):
    def __init__(self, cgroup_dir: str):
        super().__init__(version=CGROUP_V1_NAME, base_cgroup_dir=cgroup_dir)

    def read_cpu_usage_ns(self, cpu_usage_file_path: str) -> int:
        try:
            with open(cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def get_cpu_usage_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV1.CPU_ACCT_USAGE_FILE_NAME)

    def get_memory_usage_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV1.MEMORY_USAGE_FILE_NAME)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV1.MEMORY_MAX_FILE_NAME)
