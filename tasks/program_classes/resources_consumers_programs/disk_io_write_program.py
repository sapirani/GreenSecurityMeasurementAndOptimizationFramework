import os
from typing import Optional

from tasks.program_classes.abstract_program import ProgramInterface


class DiskIOWriteConsumer(ProgramInterface):
    def __init__(self, directory_path: str, number_of_files: Optional[int]):
        super().__init__()
        self.__directory_path = directory_path
        self.__number_of_files = number_of_files

    def get_program_name(self):
        return "IO Write Dummy"

    def get_command(self) -> str:
        return rf"python {os.path.join('tasks/resources_consumers', 'disk_io_writer_task.py')} -d {self.__directory_path} -n {self.__number_of_files}"
