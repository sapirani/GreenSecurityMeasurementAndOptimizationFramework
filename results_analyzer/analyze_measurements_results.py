import os

from containers_control.scanner_results_zipper import RESULTS_DIR_PREFIX
from results_analyzer.analyzer_constants import RELEVANT_PROCESSES
from results_analyzer.graph_generator import GraphsGenerator
from results_analyzer.paths_extractors.abstract_path_extractor import PathExtractor
from results_analyzer.paths_extractors.vm_path_extractor import VMPathExtractor

results_main_dir = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\project_code\Results"
graphs_output_dir_name = fr"graphs"



def print_measurement_results(measurement_results_dir: str, container_name: str) -> None:
    graphs_generator = GraphsGenerator(measurement_dir=measurement_results_dir, container_name=container_name)
    graphs_generator.display_battery_graphs()
    graphs_generator.display_cpu_graphs()
    graphs_generator.display_memory_graphs()
    graphs_generator.display_disk_io_graphs()
    graphs_generator.display_processes_graphs(RELEVANT_PROCESSES)


def print_results_graphs_per_container(container_results_dir: str, paths_extractor: PathExtractor):
    container_name = paths_extractor.get_container_name_from_path(container_results_dir)

    print(f"****** Printing {container_name} Results: ******\n\n")
    measurements_results_paths = paths_extractor.read_all_measurements_dirs(container_results_dir)
    for measurement_dir in measurements_results_paths:
        print_measurement_results(measurement_dir, container_name)


def print_results_graphs(results_dir: str, paths_extractor: PathExtractor):
    container_results_dirs = paths_extractor.read_all_containers_results_dirs(results_dir)
    for container_results_dir in container_results_dirs:
        print_results_graphs_per_container(container_results_dir, paths_extractor)


def main():
    paths_extractor = VMPathExtractor()
    results_dirs = paths_extractor.read_all_results_dirs(results_main_dir)
    for results_dir in results_dirs:
        print_results_graphs(results_dir, paths_extractor)


if __name__ == "__main__":
    main()
