import argparse
import os
import subprocess
import sys

from human_id import generate_id

DEFAULT_NUMBER_OF_EXPERIMENTS = 3

def run_scanner(session_id: str):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    scanner_path = os.path.join(repo_root, "scanner.py")

    subprocess.run([sys.executable, scanner_path, "--measurement_session_id", session_id], check=True)


def automate_experiments(num_of_experiments: int, main_session_id: str):
    for experiment_id in range(num_of_experiments):
        current_session = f"{main_session_id}_{experiment_id}"
        run_scanner(current_session)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program automates execution of scanner experiments one by one."
    )

    parser.add_argument("-n", "--number_of_experiments",
                        type=int,
                        default=DEFAULT_NUMBER_OF_EXPERIMENTS,
                        help="number of repetitions on the experiment.")


    parser.add_argument("-s", "--session_id",
                        type=str,
                        default=generate_id(word_count=3),
                        help="prefix for session_id for all measurements.")

    arguments = parser.parse_args()
    automate_experiments(arguments.number_of_experiments, arguments.session_id)

