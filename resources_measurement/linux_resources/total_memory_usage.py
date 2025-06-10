import os

import psutil

from resources_measurement.linux_resources.cgroup_utils import extract_cgroup_relative_path, detect_cgroup_version
from resources_measurement.linux_resources.config import FileKeywords
from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader

# Constants for paths and identifiers
MEMORY_USAGE_FILE_NAME_V2 = "memory.current"
MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"

MEMORY_MAX_FILE_NAME_V2 = "memory.max"
MEMORY_MAX_FILE_NAME_V1 = "memory/memory.limit_in_bytes"

HOST_PROC_MEMORY_LIMIT_FILE = r"/proc/meminfo"
MEM_TOTAL_PREFIX = "MemTotal:"


class LinuxContainerMemoryReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__memory_limit = self.__get_memory_limit_bytes(self._version)

    def get_memory_usage_bytes(self) -> int:
        try:
            with open(self._resource_usage_path) as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Error reading memory usage: {e}")
            return 0

    def get_memory_usage_percent(self) -> float:
        usage = self.get_memory_usage_bytes()
        if self.__memory_limit <= 0:
            return 0.0  # TODO: check if this is the right value
        return (usage / self.__memory_limit) * 100

    def _get_resource_file_path(self, version: str) -> str:
        cgroup_dir = extract_cgroup_relative_path(version, FileKeywords.CGROUP_V1_MEMORY_IDENTIFIER)
        if version == FileKeywords.V2:
            return os.path.join(cgroup_dir, MEMORY_USAGE_FILE_NAME_V2)
        else:
            return os.path.join(cgroup_dir, MEMORY_USAGE_FILE_NAME_V1)

    def __get_memory_limit_file(self, version: str) -> str:
        cgroup_dir = extract_cgroup_relative_path(version, FileKeywords.CGROUP_V1_MEMORY_IDENTIFIER)
        if version == FileKeywords.V2:
            return os.path.join(cgroup_dir, MEMORY_MAX_FILE_NAME_V2)
        else:
            return os.path.join(cgroup_dir, MEMORY_MAX_FILE_NAME_V1)

    def __get_host_memory_limit(self) -> int:
        try:
            return psutil.virtual_memory().total
        except Exception as e:
            print(f"Error reading host memory from /proc/meminfo: {e}")
        return 1  # Avoid division by zero

    def __get_memory_limit_bytes(self, version: str) -> int:
        max_memory_file_path = self.__get_memory_limit_file(version)
        try:
            with open(max_memory_file_path) as max_memory_file:
                max_val = max_memory_file.read().strip()
                if max_val == "max":
                    # No limit in cgroup v2, fallback to host total memory
                    limit = self.__get_host_memory_limit()
                else:
                    limit = int(max_val)

            if limit >= 2 ** 60:
                limit = self.__get_host_memory_limit()

            return limit
        except ValueError as e:
            print(f"Unexpected format in memory limit file in path {max_memory_file_path}: {max_val}")
            return self.__get_host_memory_limit()
