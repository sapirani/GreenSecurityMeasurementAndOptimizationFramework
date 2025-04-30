import os

from tasks.program_classes.abstract_program import ProgramInterface


class DummyAntivirusProgram(ProgramInterface):
    def __init__(self, scan_path):
        super().__init__()
        self.scan_path = scan_path

    def get_program_name(self):
        return "Dummy Antivirus"

    def get_command(self) -> str:
        return f"python {os.path.join('tasks/DummyPrograms', 'FilesReader.py')} {self.scan_path}"

    def general_information_before_measurement(self, f):
        f.write(f'Scan Path: {self.scan_path}\n\n')
