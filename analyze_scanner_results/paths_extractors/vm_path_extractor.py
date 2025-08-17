import os
from pathlib import Path
from typing import List

from utils.general_consts import MEASUREMENT_NAME_DIR
from analyze_scanner_results.analyzer_utils import is_results_dir
from analyze_scanner_results.paths_extractors.abstract_path_extractor import PathExtractor

RESULTS_DIR_PREFIX = "results_"
NUM_OF_CHARS_RESULT_DIR = len(RESULTS_DIR_PREFIX)

class VMPathExtractor(PathExtractor):
    def read_all_results_dirs(self, all_results_dir: str) -> List[str]:
        if os.path.exists(all_results_dir):
            return [os.path.join(all_results_dir, result_dir) for result_dir in os.listdir(all_results_dir)
                    if is_results_dir(result_dir)]

        else:
            return []

    def read_all_containers_results_dirs(self, results_main_dir: str) -> List[str]:
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


    def get_container_name_from_path(self, container_results_dir: str) -> str:
        return  os.path.basename(container_results_dir)[NUM_OF_CHARS_RESULT_DIR:]