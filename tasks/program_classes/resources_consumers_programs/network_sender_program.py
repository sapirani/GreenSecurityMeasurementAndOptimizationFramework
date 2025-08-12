import os
from typing import Optional

from utils.general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class NetworkSender(ProgramInterface):
    def __init__(self, rate: float, packet_size: Optional[int]):
        super().__init__()
        self.__packet_size = packet_size
        self.__rate = rate

    def general_information_before_measurement(self, f):
        f.write(f"Network Sender - rate: {self.__rate}\n\n")

    def get_program_name(self):
        return "Network Sender"

    def get_command(self) -> str:
        command = fr"python {os.path.join('tasks/resources_consumers', 'network_sender_task.py')} -r {self.__rate}"
        if self.__packet_size is not None:
            command += f" -s {self.__packet_size}"
        return command
