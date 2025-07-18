import os

from utils.general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class CPUConsumer(ProgramInterface):
    def __init__(self, cpu_percent_to_consume, running_time):
        super().__init__()
        self.cpu_percent_to_consume = cpu_percent_to_consume
        if running_time is None:
            self.running_time = 10 * MINUTE
        else:
            self.running_time = running_time

    def get_program_name(self):
        return "CPU Consumer"

    def get_command(self) -> str:
        return rf"python {os.path.join('tasks/DummyPrograms', 'CPUConsumer.py')} {self.cpu_percent_to_consume} {self.running_time}"
