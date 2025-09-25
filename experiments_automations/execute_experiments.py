import argparse
import os
import subprocess
import sys
from time import sleep

from human_id import generate_id

DEFAULT_NUMBER_OF_EXPERIMENTS = 3
SLEEPING_TIME_BETWEEN_MEASUREMENTS = 30

SCANNER_PROGRAM_FILE = "scanner.py"
SESSION_ID_SCANNER_FLAG = "--measurement_session_id"

SCANNER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), SCANNER_PROGRAM_FILE)


def run_scanner(scanner_path: str, session_id: str):
    subprocess.run([sys.executable, scanner_path, SESSION_ID_SCANNER_FLAG, session_id], check=True)


def automate_experiments(num_of_experiments: int, main_session_id: str):
    for experiment_id in range(num_of_experiments):
        current_session = f"{main_session_id}_{experiment_id}"
        run_scanner(SCANNER_PATH, current_session)
        sleep(SLEEPING_TIME_BETWEEN_MEASUREMENTS)


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
