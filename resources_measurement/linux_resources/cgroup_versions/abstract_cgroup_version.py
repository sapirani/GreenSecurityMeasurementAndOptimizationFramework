import os
from abc import ABC, abstractmethod

from resources_measurement.linux_resources.cgroup_versions.cgroup_v1 import CgroupV1
from resources_measurement.linux_resources.cgroup_versions.cgroup_v2 import CgroupV2
from resources_measurement.linux_resources.config import CGROUP_CONTROLLERS_FILE_PATH, CGROUP_IN_CONTAINER_PATH, \
    ProcCgroupFileConsts, FileKeywords, SYSTEM_CGROUP_FILE_PATH


class CgroupVersion(ABC):
    def __init__(self, version: str, cgroup_identifiers: str):
        self.__version_type = version
        self._cgroup_dir = extract_cgroup_relative_path(version, cgroup_identifiers)

    def get_version(self) -> str:
        return self.__version_type

    @abstractmethod
    def read_cpu_usage_n(self, cpu_usage_file_path: str) -> int:
        pass

    @abstractmethod
    def get_cpu_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_memory_usage_path(self) -> str:
        pass

    @abstractmethod
    def get_memory_limit_path(self) -> str:
        pass
