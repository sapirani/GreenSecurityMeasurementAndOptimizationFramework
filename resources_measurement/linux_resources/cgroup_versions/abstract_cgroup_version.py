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


class CgroupMetricReader(ABC):
    def __init__(self):
        self._base_cgroup_dir = self.__get_cgroup_base_dir()
        self._cpu_usage_file_path = self._get_cpu_usage_path()

    @abstractmethod
    def get_version(self) -> str:
        pass

    @abstractmethod
    def is_container_memory_limited(self, limit: str) -> bool:
        pass

    def __get_cgroup_base_dir(self) -> str:
        return SYSTEM_CGROUP_DIR_PATH

    @abstractmethod
    def read_cpu_usage_ns(self) -> int:
        pass

    @abstractmethod
    def _get_cpu_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_container_vcores(self) -> float:
        pass

    @abstractmethod
    def get_memory_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_memory_limit_path(self) -> str:
        pass
