import argparse

from time import sleep
from typing import Optional

from human_id import generate_id

from config import PROGRAM_PARAMETERS_PATH, DEFAULT_TASKS, SCANNER_PATH, SLEEPING_TIME_BETWEEN_MEASUREMENTS, \
    SLEEPING_TIME_BETWEEN_TASKS, DEFAULT_NUMBER_OF_EXPERIMENTS
from experiments_automations.run_exeperiment_utils import update_main_program, \
    update_dummy_task_values, run_identical_experiments


def run_various_experiments(num_of_experiments: int, main_session_id: str, rate: Optional[float], size: Optional[int]):
    update_dummy_task_values(PROGRAM_PARAMETERS_PATH, rate=rate, size=size)
    session_id_addition = ""
    if size is not None:
        session_id_addition += f"_{size}_bytes"
    if rate is not None:
        session_id_addition += f"_{rate}_rate"

    for task_id, task in enumerate(DEFAULT_TASKS):
        task_session = f"{main_session_id}_task_{task.name}{session_id_addition}"
        update_main_program(PROGRAM_PARAMETERS_PATH, main_program_value=task)
        run_identical_experiments(SCANNER_PATH, SLEEPING_TIME_BETWEEN_MEASUREMENTS,
                                  num_of_experiments, task_session)
        sleep(SLEEPING_TIME_BETWEEN_TASKS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program automates execution of scanner experiments one by one."
    )

    parser.add_argument("-n", "--number_of_repetitions_per_experiment",
                        type=int,
                        default=DEFAULT_NUMBER_OF_EXPERIMENTS,
                        help="number of repetitions of a single experiment.")

    parser.add_argument("-d", "--run_default_tasks",
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
    measurement_session_id = arguments.measurement_session_id

    if arguments.run_default_tasks:
        run_various_experiments(number_of_experiments, measurement_session_id, arguments.task_rate,
                                arguments.task_unit_size)

    else:
        run_identical_experiments(SCANNER_PATH, SLEEPING_TIME_BETWEEN_MEASUREMENTS,
                                  number_of_experiments, measurement_session_id)
