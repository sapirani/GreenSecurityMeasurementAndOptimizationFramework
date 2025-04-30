import os

from tasks.program_classes.abstract_program import ProgramInterface


class IOWriteConsumer(ProgramInterface):
    def __init__(self, directory_path):
        super().__init__()
        self.directory_path = directory_path

    def get_program_name(self):
        return "IO Write Dummy"

    def get_command(self) -> str:
        return f"python {os.path.join('FilesCreators', 'file_generator.py')} {self.directory_path}"
