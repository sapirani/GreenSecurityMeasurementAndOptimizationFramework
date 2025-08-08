import os

from tasks.program_classes.abstract_program import ProgramInterface


class DiskIOReadConsumer(ProgramInterface):
    def __init__(self, directory_path: str):
        super().__init__()
        self.__directory_path = directory_path

    def get_program_name(self):
        return "IO Read Dummy"

    def get_command(self) -> str:
        return rf"python {os.path.join('tasks/resources_consumers', 'disk_io_reader_task.py')} -d {self.__directory_path}"
