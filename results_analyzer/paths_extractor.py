import os
from pathlib import Path
from typing import List

from general_consts import MEASUREMENT_NAME_DIR
from results_analyzer.analyzer_utils import is_results_dir


class PathsExtractor:
    @staticmethod
    def read_all_results_dirs(all_results_dir: str) -> List[str]:
        if os.path.exists(all_results_dir):
            return [os.path.join(all_results_dir, result_dir) for result_dir in os.listdir(all_results_dir)
                    if
                    is_results_dir(result_dir)]

        else:
            return []

    @staticmethod
    def read_all_containers_results_dirs(results_main_dir: str) -> List[str]:
        containers_paths = []
        if os.path.exists(results_main_dir):
            for container_results_dir in os.listdir(results_main_dir):
                if is_results_dir(container_results_dir):
                    containers_paths.append(os.path.join(results_main_dir, container_results_dir))

        return containers_paths

    def read_all_measurements_dirs(self, container_results_dir: str) -> List[str]:
        root_path = Path(container_results_dir)
        measurement_dirs_paths = []

        for program_to_scan_dir in root_path.iterdir():
            if program_to_scan_dir.is_dir():
                self.__search_measurement_dirs(program_to_scan_dir, measurement_dirs_paths)

        return measurement_dirs_paths

    def __search_measurement_dirs(self, current_dir, measurement_dirs_paths):
        for item in current_dir.iterdir():
            if item.is_dir():
                if item.name.startswith(MEASUREMENT_NAME_DIR):
                    if any(item.iterdir()):  # Check if not empty
                        measurement_dirs_paths.append(str(item.resolve()))
                else:
                    self.__search_measurement_dirs(item, measurement_dirs_paths)
