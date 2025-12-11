import argparse
import re

from concurrent.futures import ThreadPoolExecutor
from typing import Union

from tasks.resources_consumers.memory_consumer_task import consume_ram
from tasks.resources_consumers.disk_io_writer_task import write_files

CONSUMERS_METHODS = [consume_ram, write_files]
DEFAULT_UNIT_SIZE = 1024
NUMBER_OF_TASKS = 2

FLOAT_PATTERN = re.compile(r"\d*\.\d+|\d+\.\d*")


def parse_int_or_float(token: str):
    """Parse token as int if no decimal digits, else float."""
    if FLOAT_PATTERN.fullmatch(token):
        return float(token)
    return int(token)


def parse_number_or_list(value: str):
    if "," in value:
        tokens = [v.strip() for v in value.split(",")]
        return [parse_int_or_float(tok) for tok in tokens]

    return parse_int_or_float(value)


def run_tasks_in_parallel(rate: Union[float, list[float]], size: Union[int, list[int]]):
    all_tasks_rates = []
    if isinstance(rate, float):
        all_tasks_rates = [rate] * NUMBER_OF_TASKS
    elif isinstance(rate, list):
        if len(rate) != NUMBER_OF_TASKS:
            raise ValueError(f"Number of rate options should be equal to {NUMBER_OF_TASKS} (number of tasks)")
        all_tasks_rates = rate

    all_tasks_sizes = []
    if isinstance(size, int):
        all_tasks_sizes = [size] * NUMBER_OF_TASKS
    elif isinstance(size, list):
        if len(size) != NUMBER_OF_TASKS:
            raise ValueError(f"Number of size options should be equal to {NUMBER_OF_TASKS} (number of tasks)")
        all_tasks_sizes = size

    with ThreadPoolExecutor(max_workers=NUMBER_OF_TASKS) as executor:
        for task, r, s in zip(CONSUMERS_METHODS, all_tasks_rates, all_tasks_sizes):
            executor.submit(task, r, s)


if __name__ == "__main__":
    task_description = "This program is a dummy task that consumes RAM and Disk"
    parser = argparse.ArgumentParser(description=task_description)

    parser.add_argument("-s", "--unit_size",
                        type=parse_number_or_list,
                        default=DEFAULT_UNIT_SIZE,
                        help="The size of each unit in bytes.")

    parser.add_argument("-r", "--rate",
                        type=parse_number_or_list,
                        required=True,
                        help="The number of units per second to perform the task on.")

    args = parser.parse_args()

    run_tasks_in_parallel(args.rate, args.unit_size)
