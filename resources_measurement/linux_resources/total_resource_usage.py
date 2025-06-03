from abc import ABC, abstractmethod

from resources_measurement.linux_resources.cgroup_utils import detect_cgroup_version


class LinuxContainerResourceReader(ABC):
    def __init__(self):
        self._version = detect_cgroup_version()
        self._resource_usage_path = self._get_resource_file_path(self._version)

    @abstractmethod
    def _get_resource_file_path(self, version: str) -> str:
        pass