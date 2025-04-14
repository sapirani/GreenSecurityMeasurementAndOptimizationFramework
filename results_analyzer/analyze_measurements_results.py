import os
from pathlib import Path
from typing import List

from containers_control.scanner_results_zipper import RESULTS_DIR_PREFIX
from general_consts import MEASUREMENT_NAME_DIR
from results_analyzer.analyzer_utils import is_results_dir
from results_analyzer.graph_generator import GraphsGenerator

results_main_dir = fr"C:\Users\Administrator\Desktop\GreenSecurityAll\results"
graphs_output_dir_name = fr"graphs"

NUM_OF_CHARS_RESULT_DIR = len(RESULTS_DIR_PREFIX)



def read_all_results_dirs(main_result_dir: str) -> List[str]:
    if os.path.exists(main_result_dir):
        return [os.path.join(main_result_dir, result_dir) for result_dir in os.listdir(main_result_dir) if
                is_results_dir(result_dir)]

    else:
        return []


def search_measurement_dirs(current_dir, measurement_dirs_paths):
    for item in current_dir.iterdir():
        if item.is_dir():
            if item.name.startswith(MEASUREMENT_NAME_DIR):
                if any(item.iterdir()):  # Check if not empty
                    measurement_dirs_paths.append(str(item.resolve()))
            else:
                search_measurement_dirs(item, measurement_dirs_paths)


def read_all_measurements_dirs(root_dir):
    root_path = Path(root_dir)
    measurement_dirs_paths = []

    for program_to_scan_dir in root_path.iterdir():
        if program_to_scan_dir.is_dir():
            search_measurement_dirs(program_to_scan_dir, measurement_dirs_paths)

    return measurement_dirs_paths


def print_container_name(container_results_dir: str) -> str:
    container_name = os.path.basename(container_results_dir)[NUM_OF_CHARS_RESULT_DIR:]
    print(f"****** Printing {container_name} Results: ******\n\n")
    return container_name


def print_measurement_results(measurement_results_dir: str, container_name: str) -> None:
    graphs_generator = GraphsGenerator(measurement_dir=measurement_results_dir, container_name=container_name)
    # graphs_generator.display_battery_graphs()
    # graphs_generator.display_cpu_graphs()
    # graphs_generator.display_memory_graphs()
    # graphs_generator.display_disk_io_graphs()
    graphs_generator.display_processes_graphs([1214])


def print_results_graphs_per_container(container_results_dir: str):
    container_name = print_container_name(container_results_dir)
    measurements_results_paths = read_all_measurements_dirs(container_results_dir)
    for measurement_dir in measurements_results_paths:
        print_measurement_results(measurement_dir, container_name)


def print_results_graphs(results_dir: str):
    if os.path.exists(results_dir):
        for container_results_dir in os.listdir(results_dir):
            if is_results_dir(container_results_dir):
                print_results_graphs_per_container(os.path.join(results_dir, container_results_dir))


def main():
    results_dirs = read_all_results_dirs(results_main_dir)
    for results_dir in results_dirs:
        print_results_graphs(results_dir)


if __name__ == "__main__":
    main()
