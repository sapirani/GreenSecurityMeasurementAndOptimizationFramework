from abc import ABC, abstractmethod


class CgroupVersion(ABC):
    def __init__(self, version: str, base_cgroup_dir: str):
        self.__version_type = version
        self.__base_cgroup_dir = base_cgroup_dir

    def get_version(self) -> str:
        return self.__version_type

    @abstractmethod
    def read_cpu_usage_ns(self, cpu_usage_file_path: str) -> int:
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
