import argparse
import re

from concurrent.futures import ThreadPoolExecutor
from typing import Union, Callable

from tasks.resources_consumers.memory_consumer_task import consume_ram
from tasks.resources_consumers.memory_releaser_task import release_ram
from tasks.resources_consumers.disk_io_writer_task import write_files
from tasks.resources_consumers.disk_io_reader_task import read_files
from tasks.resources_consumers.network_sender_task import send_udp_packets
from tasks.resources_consumers.network_receiver_task import receive_udp_packets

CONSUMERS_METHODS = [consume_ram, write_files]
DEFAULT_UNIT_SIZE = 1024

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


def run_tasks_in_parallel(tasks_to_run: list[Callable], rate: Union[float, list[float]], size: Union[int, list[int]]):
    number_of_tasks = len(tasks_to_run)
    all_tasks_rates = []
    if isinstance(rate, float):
        all_tasks_rates = [rate] * number_of_tasks
    elif isinstance(rate, list):
        if len(rate) != number_of_tasks:
            raise ValueError(f"Number of rate options should be equal to {number_of_tasks} (number of tasks)")
        all_tasks_rates = rate

    all_tasks_sizes = []
    if isinstance(size, int):
        all_tasks_sizes = [size] * number_of_tasks
    elif isinstance(size, list):
        if len(size) != number_of_tasks:
            raise ValueError(f"Number of size options should be equal to {number_of_tasks} (number of tasks)")
        all_tasks_sizes = size

    with ThreadPoolExecutor(max_workers=number_of_tasks) as executor:
        for task, r, s in zip(tasks_to_run, all_tasks_rates, all_tasks_sizes):
            executor.submit(task, r, s)


if __name__ == "__main__":
    # todo: think about supporting CPU consumer (since it creates new process)
    task_description = "This program is a dummy task that consumes RAM, Disk and Network"
    parser = argparse.ArgumentParser(description=task_description)

    parser.add_argument("-s", "--unit_size",
                        type=parse_number_or_list,
                        default=DEFAULT_UNIT_SIZE,
                        help="The size of each unit in bytes.")

    parser.add_argument("-r", "--rate",
                        type=parse_number_or_list,
                        required=True,
                        help="The number of units per second to perform the task on.")

    parser.add_argument("--run_memory_consumer",
                        type=bool,
                        default=False,
                        help="Whether to run the memory consumer task.")

    parser.add_argument("--run_memory_releaser",
                        type=bool,
                        default=False,
                        help="Whether to run the memory releaser task.")

    parser.add_argument("--run_disk_writer",
                        type=bool,
                        default=False,
                        help="Whether to run the disk writer task.")

    parser.add_argument("--run_disk_reader",
                        type=bool,
                        default=False,
                        help="Whether to run the disk reader task.")

    parser.add_argument("--run_network_receiver",
                        type=bool,
                        default=False,
                        help="Whether to run the network receiver task.")

    parser.add_argument("--run_network_sender",
                        type=bool,
                        default=False,
                        help="Whether to run the network sender task.")


    args = parser.parse_args()
    tasks_to_run = []

    if args.run_memory_consumer:
        tasks_to_run.append(consume_ram)

    if args.run_memory_releaser:
        tasks_to_run.append(release_ram)

    if args.run_disk_writer:
        tasks_to_run.append(write_files)

    if args.run_disk_reader:
        tasks_to_run.append(read_files)

    if args.run_network_receiver:
        tasks_to_run.append(receive_udp_packets)

    if args.run_network_sender:
        tasks_to_run.append(send_udp_packets)

    run_tasks_in_parallel(tasks_to_run, args.rate, args.unit_size)
