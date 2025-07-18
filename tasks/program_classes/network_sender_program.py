import os

from utils.general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class NetworkSender(ProgramInterface):
    def __init__(self, time_interval, running_time):
        super().__init__()
        self.time_interval = time_interval
        if running_time is None:
            self.running_time = 10 * MINUTE
        else:
            self.running_time = running_time

    def general_information_before_measurement(self, f):
        f.write(f"Network Sender - time interval: {self.time_interval}\n\n")

    def get_program_name(self):
        return "Network Sender"

    def get_command(self) -> str:
        return fr"python {os.path.join('tasks/DummyPrograms', 'DummyNetworkSender.py')} {self.running_time} {self.time_interval}"
