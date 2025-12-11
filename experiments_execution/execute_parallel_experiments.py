import argparse
import itertools

from time import sleep
from typing import Optional

from human_id import generate_id

from utils.general_consts import ProgramToScan
from config import DEFAULT_NUMBER_OF_EXPERIMENTS, SLEEPING_TIME_BETWEEN_MEASUREMENTS, SLEEPING_TIME_BETWEEN_TASKS, \
    SCANNER_PATH, PROGRAM_PARAMETERS_PATH, DEFAULT_TASKS
from experiments_execution.run_exeperiment_utils import update_main_program, \
    update_dummy_task_values, run_identical_experiments, update_background_programs


def run_various_experiments(tasks_to_run: list[ProgramToScan], num_of_experiments: int, num_of_parallel_tasks: int,
                            main_session_id: str, rate: Optional[float], size: Optional[int]):
    update_dummy_task_values(PROGRAM_PARAMETERS_PATH, rate=rate, size=size)
    session_id_addition = ""
    if size is not None:
        session_id_addition += f"_{size}_bytes"
    if rate is not None:
        session_id_addition += f"_{rate}_rate"

    possible_tasks_combinations = list(itertools.combinations(tasks_to_run, num_of_parallel_tasks))

    for task_id, task_combination in enumerate(possible_tasks_combinations):
        main_program = task_combination[0]
        background_program = list(task_combination[1:])
        update_main_program(PROGRAM_PARAMETERS_PATH, main_program_value=main_program)
        update_background_programs(PROGRAM_PARAMETERS_PATH, background_programs_value=background_program)

        background_tasks_names = "_".join([t.name for t in background_program])
        task_session = f"{main_session_id}_m_{main_program.name}_b_{background_tasks_names}{session_id_addition}"
        run_identical_experiments(SCANNER_PATH, SLEEPING_TIME_BETWEEN_MEASUREMENTS,
                                  num_of_experiments, task_session)
        sleep(SLEEPING_TIME_BETWEEN_TASKS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program automates execution of scanner experiments in parallel."
    )

    parser.add_argument("-n", "--number_of_repetitions_per_experiment",
                        type=int,
                        default=DEFAULT_NUMBER_OF_EXPERIMENTS,
                        help="number of repetitions of a single experiment.")

    parser.add_argument("-t", "--number_of_parallel_tasks",
                        type=int,
                        default=-1,
                        help="number of parallel tasks in a single experiment.")

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

    parser.add_argument("--run_memory_consumer",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the memory consumer task.")

    parser.add_argument("--run_memory_releaser",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the memory releaser task.")

    parser.add_argument("--run_disk_writer",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the disk writer task.")

    parser.add_argument("--run_disk_reader",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the disk reader task.")

    parser.add_argument("--run_cpu_consumer",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the cpu consumer task.")

    parser.add_argument("--run_network_receiver",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the network receiver task.")

    parser.add_argument("--run_network_sender",
                        type=bool,
                        action="store_true",
                        default=False,
                        help="Whether to run the network sender task.")

    arguments = parser.parse_args()

    number_of_experiments = arguments.number_of_repetitions_per_experiment
    number_of_parallel_tasks = arguments.number_of_parallel_tasks
    measurement_session_id = arguments.measurement_session_id

    tasks_to_run = []
    if arguments.run_memory_consumer:
        tasks_to_run.append(ProgramToScan.MemoryConsumer)

    if arguments.run_memory_releaser:
        tasks_to_run.append(ProgramToScan.MemoryReleaser)

    if arguments.run_disk_writer:
        tasks_to_run.append(ProgramToScan.DiskIOWriteConsumer)

    if arguments.run_disk_reader:
        tasks_to_run.append(ProgramToScan.DiskIOReadConsumer)

    if arguments.run_cpu_consumer:
        tasks_to_run.append(ProgramToScan.CPUConsumer)

    if arguments.run_network_receiver:
        tasks_to_run.append(ProgramToScan.NetworkReceiver)

    if arguments.run_network_sender:
        tasks_to_run.append(ProgramToScan.NetworkSender)

    if len(tasks_to_run) == 0:
        tasks_to_run = DEFAULT_TASKS

    if number_of_parallel_tasks == -1:
        number_of_parallel_tasks = len(tasks_to_run)

    run_various_experiments(tasks_to_run, number_of_experiments, number_of_parallel_tasks,
                            measurement_session_id, arguments.task_rate,
                            arguments.task_unit_size)
