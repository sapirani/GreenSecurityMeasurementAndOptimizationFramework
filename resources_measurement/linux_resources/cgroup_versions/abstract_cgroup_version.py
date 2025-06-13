import os
from abc import ABC, abstractmethod

from resources_measurement.linux_resources.cgroup_versions.cgroup_entry import CgroupEntry
from resources_measurement.linux_resources.cgroup_versions.common_paths import SYSTEM_CGROUP_DIR_PATH

# Contains details on the cgroup of the container.
# The file format is one or more lines, where each line indicates a hierarchy, its controllers, and the cgroup path for the process:
# hierarchy-ID : controllers : cgroup-path -> WHERE:
# Hierarchy ID can be 0 (for cgroup v2), or 2, 3, etc. in v1
# Controller(s)	can be cpu,cpuacct or memory for v1 or empty string ("") for v2
# Path to cgroup can be /docker/<container-id> or /
CGROUP_TYPE_PATH = r"/proc/self/cgroup"


class ProcCgroupFileConsts:
    NUMBER_OF_ELEMENTS = 3
    HIERARCHY_INDEX = 0
    CONTROLLERS_INDEX = 1
    CGROUP_PATH_INDEX = 2


class CgroupVersion(ABC):
    def __init__(self):
        self._base_cgroup_dir = self._get_cgroup_base_dir()
        self._cpu_usage_file_path = self._get_cpu_usage_path()

    @abstractmethod
    def get_version(self) -> str:
        pass

    @abstractmethod
    def _is_cgroup_dir(self, cgroup_entry: CgroupEntry) -> bool:
        pass

    @abstractmethod
    def is_container_memory_limited(self, limit: str) -> bool:
        pass

    def _get_cgroup_base_dir(self) -> str:
        with open(CGROUP_TYPE_PATH, "r") as f:
            for line in f:
                cgroup_entry = CgroupEntry.from_line(line)
                if self._is_cgroup_dir(cgroup_entry):
                    return os.path.join(SYSTEM_CGROUP_DIR_PATH, cgroup_entry.cgroup_path)

        return SYSTEM_CGROUP_DIR_PATH

    @abstractmethod
    def read_cpu_usage_ns(self) -> int:
        pass

    @abstractmethod
    def _get_cpu_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_memory_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_memory_limit_path(self) -> str:
        pass
