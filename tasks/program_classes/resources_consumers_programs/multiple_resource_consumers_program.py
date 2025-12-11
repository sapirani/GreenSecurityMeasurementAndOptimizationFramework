import os
from typing import Union

from tasks.program_classes.abstract_program import ProgramInterface


class MultipleResourceConsumer(ProgramInterface):
    def __init__(self, tasks_to_run: list[str], consumption_speed: Union[float, list[float]], chunk_size: Union[int, list[int]]):
        super().__init__()
        self.__chunk_size = chunk_size
        self.__consumption_speed = consumption_speed
        self.__tasks_to_run = tasks_to_run

    def general_information_before_measurement(self, f):
        f.write(f"Multiple Resource Consumer - chunk size: {self.__chunk_size} bytes,"
                f" speed: {self.__consumption_speed}\n\n")

    def get_program_name(self):
        return "Multiple Resource Consumer"

    def get_command(self) -> str:
        rate_str = ",".join([f"{r}" for r in self.__consumption_speed]) if isinstance(self.__consumption_speed, list) else f"{self.__consumption_speed}"
        size_str = ",".join([f"{s}" for s in self.__chunk_size]) if isinstance(self.__chunk_size, list) else f"{self.__chunk_size}"
        command = f"python {os.path.join('tasks/resources_consumers', 'multiple_resource_consumers_task.py')} -r {rate_str} -s {size_str}"
        for task in self.__tasks_to_run:
            command += f" --{task}"

        return command
