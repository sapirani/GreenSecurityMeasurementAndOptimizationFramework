import os

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion

MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"
MEMORY_MAX_FILE_NAME_V1 = "memory/memory.limit_in_bytes"
# Reports the total CPU time consumed by tasks in the cgroup. - relevant for cgroup V1
# The format of the file is a single integer value representing nanoseconds.
CPU_ACCT_USAGE_FILE_NAME = r"cpuacct.usage"

CGROUP_V1_CPU_IDENTIFIER = "cpu,cpuacct"


class CgroupV1(CgroupVersion):
    def __init__(self):
        super().__init__(version="V1", cgroup_identifiers=CGROUP_V1_CPU_IDENTIFIER)

    def read_cpu_usage_n(self, cpu_usage_file_path: str) -> int:
        try:
            with open(cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def get_cpu_usage_path(self) -> str:
        return os.path.join(self._cgroup_dir, CPU_ACCT_USAGE_FILE_NAME)

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._cgroup_dir, MEMORY_USAGE_FILE_NAME_V1)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._cgroup_dir, MEMORY_MAX_FILE_NAME_V1)
