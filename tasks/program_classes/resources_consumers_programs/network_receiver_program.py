import os
from typing import Optional

from tasks.program_classes.abstract_program import ProgramInterface


class NetworkReceiver(ProgramInterface):
    def __init__(self, rate: float, buffer_size: Optional[int]):
        super().__init__()
        self.__buffer_size = buffer_size
        self.__rate = rate

    def get_program_name(self):
        return "Network Receiver"

    def get_command(self) -> str:
        command = fr"python {os.path.join('tasks/resources_consumers', 'network_receiver_task.py')} -r {self.__rate}"
        if self.__buffer_size is not None:
            command += f" -s {self.__buffer_size}"
        return command
