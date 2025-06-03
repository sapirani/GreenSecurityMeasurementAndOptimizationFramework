import os

from resources_measurement.linux_resources.cgroup_utils import extract_cgroup_relative_path, detect_cgroup_version
from resources_measurement.linux_resources.config import FileKeywords
from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader

# Constants for paths and identifiers
MEMORY_USAGE_FILE_NAME_V2 = "memory.current"
MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"


class LinuxContainerMemoryReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()

    def get_memory_usage_bytes(self) -> int:
        try:
            with open(self._resource_usage_path) as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Error reading memory usage: {e}")
            return 0

    def _get_resource_file_path(self, version: str) -> str:
        cgroup_dir = extract_cgroup_relative_path(version, FileKeywords.CGROUP_V1_MEMORY_IDENTIFIER)
        if version == FileKeywords.V2:
            return os.path.join(cgroup_dir, MEMORY_USAGE_FILE_NAME_V2)
        else:
            return os.path.join(cgroup_dir, MEMORY_USAGE_FILE_NAME_V1)
