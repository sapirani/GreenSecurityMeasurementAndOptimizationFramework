import os
import subprocess
import zipfile
import glob
from typing import List

PROJECT_NUMBER = "e39ba8b9-65fb-4ecd-aaca-be2e4d72c868"
DOCKER_PATH = fr"/opt/gns3/projects/{PROJECT_NUMBER}/project-files/docker"
VOLUME_MAIN_DIRECTORY = r"green_security_measurements"
SCANNER_DIRECTORY = r"Scanner"
RESULTS_DIR_NAME = "results_"

OUTPUT_ZIP_NAME = "./all_results.zip"
MAIN_CONTAINERS = ["resourcemanager-1", "namenode-1", "historyserver-1"]
AVAILABLE_DATANODES = ["datanode1-1", "datanode2-1", "datanode3-1"]
ALL_NODES = MAIN_CONTAINERS + AVAILABLE_DATANODES

NUM_OF_CONTAINERS = 5
COMMAND_FOR_VOLUMES_LIST = f'ls -lat | awk \'$9 != "" {{print $9}}\' | head -n {NUM_OF_CONTAINERS}'


def get_volumes_directories() -> List[str]:
    volumes_res = subprocess.run(COMMAND_FOR_VOLUMES_LIST, shell=True, capture_output=True, text=True, cwd=DOCKER_PATH)
    output_lines = volumes_res.stdout.strip().split("\n")
    if isinstance(output_lines, list) and all(isinstance(line, str) for line in output_lines) and len(
            output_lines) == NUM_OF_CONTAINERS:
        print("Valid output:", output_lines)
        return output_lines
    else:
        print("Invalid output")
        raise Exception("Invalid Volumes Names")

def get_results_names(results_paths: List[str]) -> List[str]:
    containers_with_results = []
    for res_path in results_paths:
        results_dir_name = os.path.basename(res_path)
        if results_dir_name.startswith(RESULTS_DIR_NAME):
            container_name = results_dir_name[len(RESULTS_DIR_NAME):]
            containers_with_results.append(container_name)
    print("THE CONTAINERS:", containers_with_results) 
    return containers_with_results


def get_results_paths(volumes_list: List[str]) -> List[str]:
    available_containers = MAIN_CONTAINERS + AVAILABLE_DATANODES[:NUM_OF_CONTAINERS - len(MAIN_CONTAINERS)]

    dirs_to_zip = [fr"{DOCKER_PATH}/{volume_dir}/{VOLUME_MAIN_DIRECTORY}/{SCANNER_DIRECTORY}/" for
                   volume_dir in volumes_list]

    paths_of_results = [[res_path for res_path, _, _ in os.walk(container_dir) if RESULTS_DIR_NAME in res_path] for container_dir in dirs_to_zip]
    available_result_dirs_per_volume = [results_dirs[0] for results_dirs in paths_of_results if len(results_dirs) > 0]
    print("AVAILABLE:", paths_of_results)
    existing_results = get_results_names(available_result_dirs_per_volume)

    missing_dirs = [d for d in ALL_NODES[:NUM_OF_CONTAINERS] if not d in existing_results] # check which directory actually exists
    if missing_dirs:
        print(f"Warning: These directories do not exist and will be skipped:\n{missing_dirs}")

    print(f"\n\nZipping results of: {available_result_dirs_per_volume}")
    return available_result_dirs_per_volume


def get_results_from_containers(results_dirs_list: List[str]):
    with zipfile.ZipFile(OUTPUT_ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for container_index, results_dir in enumerate(results_dirs_list):
            zip_dir_name = f"{results_dir}_{container_index}"
            for root, _, files in os.walk(results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(zip_dir_name, os.path.relpath(file_path, results_dir))
                    zipf.write(file_path, arcname)
    print(f"Zipped directories into {OUTPUT_ZIP_NAME}")


if __name__ == "__main__":
    volumes_list = get_volumes_directories()
    results_paths = get_results_paths(volumes_list)
    get_results_from_containers(results_paths)
