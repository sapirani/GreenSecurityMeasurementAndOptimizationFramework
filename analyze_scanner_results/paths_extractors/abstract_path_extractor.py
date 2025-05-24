from abc import ABC, abstractmethod
from typing import List


class PathExtractor(ABC):
    @abstractmethod
    def read_all_results_dirs(self, all_results_dir: str) -> List[str]:
        pass

    @abstractmethod
    def read_all_containers_results_dirs(self, results_main_dir: str) -> List[str]:
        pass

    @abstractmethod
    def read_all_measurements_dirs(self, container_results_dir: str) -> List[str]:
        pass

    @abstractmethod
    def get_container_name_from_path(self, container_results_dir: str) -> str:
        pass
