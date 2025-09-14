import os
from typing import Optional

from tasks.program_classes.abstract_program import ProgramInterface


class DiskIOReadConsumer(ProgramInterface):
    def __init__(self, rate: float, file_size: Optional[int]):
        super().__init__()
        self.__rate = rate
        self.__file_size = file_size

    def get_program_name(self):
        return "IO Read Dummy"

    def get_command(self) -> str:
        command = rf"python {os.path.join('tasks/resources_consumers', 'disk_io_reader_task.py')} -r {self.__rate} "
        if self.__file_size is not None:
            command += f"-s {self.__file_size}"
        return command