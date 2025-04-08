import os
import subprocess
import zipfile
from typing import List, Optional, Dict

# Configuration
NUM_OF_CONTAINERS = 5
PROJECT_NUMBER = "e39ba8b9-65fb-4ecd-aaca-be2e4d72c868"
FIRST_DOCKER_PATH = fr"/opt/gns3/projects/{PROJECT_NUMBER}/project-files/docker"
SECOND_DOCKER_PATH = r"/var/lib/docker/volumes"

VOLUME_MAIN_DIRECTORY = "green_security_measurements"
SCANNER_DIRECTORY = "Scanner"
RESULTS_DIR_PREFIX = "results_"
OUTPUT_ZIP_NAME = "./all_results.zip"

MAIN_CONTAINERS = ["resourcemanager-1", "namenode-1", "historyserver-1"]
AVAILABLE_DATANODES = [f"datanode{idx + 1}-1" for idx in range(NUM_OF_CONTAINERS - len(MAIN_CONTAINERS))]
ALL_NODES = MAIN_CONTAINERS + AVAILABLE_DATANODES


def run_command_in_dir(command: str) -> List[str]:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip().split("\n")
    if result.returncode != 0 or not output:
        raise RuntimeError(f"Failed to run command: {result.stderr}")
    return output


def find_recent_volume_dirs(count: int) -> List[str]:
    command = (
        f'docker ps -a -q | head -n {count} | xargs docker inspect --format '
        f'\'{{range.Mounts}}{{if and (eq .Type "volume") (eq .Destination /{VOLUME_MAIN_DIRECTORY})}}{{.Source}}{{"\n"}}{{end}}{{end}}\''
    )
    volume_dirs = run_command_in_dir(command)
    return volume_dirs


def find_results_dirs(volume_dirs: List[str]) -> List[str]:
    results_dirs = []
    for vol_dir in volume_dirs:
        search_path = os.path.join(vol_dir, VOLUME_MAIN_DIRECTORY, SCANNER_DIRECTORY)
        for root, dirs, _ in os.walk(search_path):
            for d in dirs:
                if d.startswith(RESULTS_DIR_PREFIX):
                    results_dirs.append(os.path.join(root, d))
                    if len(results_dirs) == NUM_OF_CONTAINERS:
                        return results_dirs
    return results_dirs


def extract_container_names(results_paths: List[str]) -> List[str]:
    return [
        os.path.basename(path)[len(RESULTS_DIR_PREFIX):]
        for path in results_paths
        if os.path.basename(path).startswith(RESULTS_DIR_PREFIX)
    ]


def extract_results_map(results_paths: List[str]) -> Dict[str, str]:
    """
    Returns a mapping from container name to the first path found for its result directory.
    """
    container_to_path = {}
    for path in results_paths:
        basename = os.path.basename(path)
        if basename.startswith(RESULTS_DIR_PREFIX):
            container = basename[len(RESULTS_DIR_PREFIX):]
            if container not in container_to_path:
                container_to_path[container] = path
    return container_to_path


def zip_directories(directories: List[str], output_path: str) -> None:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=os.path.dirname(directory))
                    zipf.write(full_path, arcname=arcname)
    print(f"Zipped output saved to: {output_path}")


def main():
    volume_dirs = find_recent_volume_dirs(NUM_OF_CONTAINERS)
    results_dirs = find_results_dirs(volume_dirs)

    if len(results_dirs) != NUM_OF_CONTAINERS:
        print("Could not find the required number of result directories across all paths.")

    found_containers = extract_container_names(results_dirs)
    expected_containers = ALL_NODES[:NUM_OF_CONTAINERS]
    missing = [c for c in expected_containers if c not in found_containers]
    if missing:
        print(f"Missing results for containers: {missing}")

    print(f"Zipping results from: {results_dirs}")
    zip_directories(results_dirs, OUTPUT_ZIP_NAME)


if __name__ == "__main__":
    main()
