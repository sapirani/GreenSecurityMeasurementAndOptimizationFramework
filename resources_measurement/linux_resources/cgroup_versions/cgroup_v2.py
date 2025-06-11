import os

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion


class CgroupV2Keywords:
    USAGE_USEC = "usage_usec"


class FilePathsV2:
    MEMORY_USAGE_FILE_NAME = "memory.current"
    MEMORY_MAX_FILE_NAME = "memory.max"

    # Provides CPU usage statistics for the cgroup.
    # The format of the file is key-value pairs, one per line.
    # E.g.
    #   usage_usec 123456789
    #   user_usec 12345678
    #   system_usec 11111111
    # The interesting key is usage_usec, its value is the total CPU time consumed by all tasks in the cgroup, in microseconds.
    CPU_STATS_FILE_NAME = "cpu.stat"


CGROUP_V2_NAME = "V2"


class CgroupV2(CgroupVersion):
    def __init__(self, cgroup_dir: str):
        super().__init__(version=CGROUP_V2_NAME, base_cgroup_dir=cgroup_dir)

    def read_cpu_usage_ns(self, cpu_usage_file_path: str) -> int:
        try:
            with open(cpu_usage_file_path) as f:
                for line in f:
                    if line.startswith(CgroupV2Keywords.USAGE_USEC):
                        return int(line.split()[1]) * 1000  # convert to nanoseconds 143512538

            return 0
        except Exception as e:
            raise ValueError(f"The file {cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def get_cpu_usage_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV2.CPU_STATS_FILE_NAME)

    def get_memory_usage_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV2.MEMORY_USAGE_FILE_NAME)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self.__base_cgroup_dir, FilePathsV2.MEMORY_MAX_FILE_NAME)
