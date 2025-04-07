import os
import subprocess
import zipfile
from typing import List, Optional, Dict

# Configuration
NUM_OF_CONTAINERS = 5
PROJECT_NUMBER = "e39ba8b9-65fb-4ecd-aaca-be2e4d72c868"
FIRST_DOCKER_PATH = fr"/opt/gns3/projects/{PROJECT_NUMBER}/project-files/docker"
SECOND_DOCKER_PATH = r"/var/lib/docker/volumes"
DATA_DIR_IN_VOLUME = r"/_data"
VOLUME_OPTIONAL_PATHS = [FIRST_DOCKER_PATH, SECOND_DOCKER_PATH]

VOLUME_MAIN_DIRECTORY = "green_security_measurements"
SCANNER_DIRECTORY = "Scanner"
RESULTS_DIR_PREFIX = "results_"
OUTPUT_ZIP_NAME = "./all_results.zip"

MAIN_CONTAINERS = ["resourcemanager-1", "namenode-1", "historyserver-1"]
AVAILABLE_DATANODES = [f"datanode{idx + 1}-1" for idx in range(NUM_OF_CONTAINERS - len(MAIN_CONTAINERS))]
ALL_NODES = MAIN_CONTAINERS + AVAILABLE_DATANODES


def run_command_in_dir(command: str, cwd: str) -> List[str]:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    output = result.stdout.strip().split("\n")
    if result.returncode != 0 or not output:
        raise RuntimeError(f"Failed to run command in {cwd}: {result.stderr}")
    return output


def find_recent_volume_dirs(base_path: str, count: int) -> List[str]:
    command = (
        f'ls -lat | awk \'$9 != "" {{print $9}}\' '
        f'| grep -v "^\\.$" | grep -v "^\\.\\.$" | grep -v "^metadata\\.db$" '
        f'| head -n {count}'
    )
    volume_dirs = run_command_in_dir(command, cwd=base_path)
    return [os.path.join(base_path, vol) for vol in volume_dirs]


def find_results_dirs(volume_dirs: List[str]) -> List[str]:
    results_dirs = []
    for vol_dir in volume_dirs:
        if vol_dir.startswith(SECOND_DOCKER_PATH):
            vol_dir = vol_dir + DATA_DIR_IN_VOLUME

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
    final_results_dirs = []

    for path in VOLUME_OPTIONAL_PATHS:
        try:
            volume_dirs = find_recent_volume_dirs(path, NUM_OF_CONTAINERS)
            results_dirs = find_results_dirs(volume_dirs)

            print(f"Found {len(results_dirs)} results in {path}")

            if len(results_dirs) == NUM_OF_CONTAINERS:
                final_results_dirs = results_dirs
                print(f"Found all results in one path. Using these only.")
                break
            else:
                final_results_dirs.extend(results_dirs)
                container_map = extract_results_map(final_results_dirs)
                if len(container_map) >= NUM_OF_CONTAINERS:
                    print(f"Aggregated enough unique containers across paths.")
                    # Take exactly NUM_OF_CONTAINERS unique container result paths
                    final_results_dirs = list(container_map.values())[:NUM_OF_CONTAINERS]
                    break
        except Exception as e:
            print(f"Error accessing path {path}: {e}")

    if len(final_results_dirs) != NUM_OF_CONTAINERS:
        raise RuntimeError("Could not find the required number of result directories across all paths.")

    found_containers = extract_container_names(final_results_dirs)
    expected_containers = ALL_NODES[:NUM_OF_CONTAINERS]
    missing = [c for c in expected_containers if c not in found_containers]
    if missing:
        print(f"Missing results for containers: {missing}")

    print(f"Zipping results from: {final_results_dirs}")
    zip_directories(final_results_dirs, OUTPUT_ZIP_NAME)


if __name__ == "__main__":
    main()
