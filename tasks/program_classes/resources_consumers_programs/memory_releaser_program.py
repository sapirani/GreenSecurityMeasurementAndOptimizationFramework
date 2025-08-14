import os
from typing import Optional

from tasks.program_classes.abstract_program import ProgramInterface


class MemoryReleaser(ProgramInterface):
    def __init__(self, releasing_speed: float, memory_chunk_size: Optional[int]):
        super().__init__()
        self.__memory_chunk_size = memory_chunk_size
        self.__releasing_speed = releasing_speed

    def general_information_before_measurement(self, f):
        f.write(f"Memory Releaser - chunk size: {self.__memory_chunk_size} bytes,"
                f" speed: {self.__releasing_speed} bytes per second\n\n")

    def get_program_name(self):
        return "Memory Releaser"

    def get_command(self) -> str:
        command = f"python {os.path.join('tasks/resources_consumers', 'memory_releaser_task.py')} -r {self.__releasing_speed}"
        if self.__memory_chunk_size is not None:
            command += f" -s {self.__memory_chunk_size}"
        return command
