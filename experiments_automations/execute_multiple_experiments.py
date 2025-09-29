import argparse
import os

from time import sleep
from typing import Optional

from human_id import generate_id

from experiments_automations.run_exeperiment_utils import run_scanner, update_main_program, \
    update_dummy_task_values
from utils.general_consts import MINUTE, ProgramToScan

DEFAULT_NUMBER_OF_EXPERIMENTS = 3
SLEEPING_TIME_BETWEEN_MEASUREMENTS = 30
SLEEPING_TIME_BETWEEN_TASKS = 2 * MINUTE
DEFAULT_TASK_UNIT_SIZE = 1024

SCANNER_PROGRAM_FILE = "scanner.py"
SCANNER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), SCANNER_PROGRAM_FILE)

PROGRAM_PARAMETERS_FILE = "program_parameters.py"
PROGRAM_PARAMETERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), PROGRAM_PARAMETERS_FILE)

DEFAULT_MAIN_PROGRAM = ProgramToScan.BASELINE_MEASUREMENT
DEFAULT_TASKS = [ProgramToScan.MemoryConsumer, ProgramToScan.MemoryReleaser,
                 ProgramToScan.DiskIOReadConsumer, ProgramToScan.DiskIOWriteConsumer,
                 ProgramToScan.CPUConsumer]


def run_identical_experiments(num_of_experiments: int, main_session_id: str):
    for experiment_id in range(num_of_experiments):
        current_session = f"{main_session_id}_{experiment_id}"
        run_scanner(SCANNER_PATH, current_session)
        sleep(SLEEPING_TIME_BETWEEN_MEASUREMENTS)


def run_various_experiments(num_of_experiments: int, rate: Optional[float], size: Optional[int]):
    update_dummy_task_values(PROGRAM_PARAMETERS_PATH, rate=rate, size=size)
    for task_id, task in enumerate(DEFAULT_TASKS):
        task_session = f"default_task_{task.name}"
        update_main_program(PROGRAM_PARAMETERS_PATH, main_program_value=task)
        run_identical_experiments(num_of_experiments, task_session)
        sleep(SLEEPING_TIME_BETWEEN_TASKS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program automates execution of scanner experiments one by one."
    )

    parser.add_argument("-n", "--number_of_repetitions_per_experiment",
                        type=int,
                        default=DEFAULT_NUMBER_OF_EXPERIMENTS,
                        help="number of repetitions of a single experiment.")

    parser.add_argument("-d", "--use_default",
                        type=bool,
                        default=False,
                        help="choose whether to run the scanner with the current program_parameters configuration, or run all default tasks.")

    parser.add_argument("-i", "--measurement_session_id",
                        type=str,
                        default=generate_id(word_count=3),
                        help="prefix for session_id for all measurements.")

    parser.add_argument("-s", "--task_unit_size",
                        type=int,
                        default=None,
                        help="The size of each unit in bytes that the default task will use.")

    parser.add_argument("-r", "--task_rate",
                        type=float,
                        default=None,
                        help="The number of units per second to perform the default task on.")

    arguments = parser.parse_args()
    number_of_experiments = arguments.number_of_repetitions_per_experiment
    measurement_session = arguments.measurement_session_id

    if arguments.use_default:
        run_various_experiments(number_of_experiments, arguments.task_rate, arguments.task_unit_size)

    else:
        run_identical_experiments(number_of_experiments, measurement_session)
