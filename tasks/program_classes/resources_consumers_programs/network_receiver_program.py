import os
from typing import Optional

from tasks.program_classes.abstract_program import ProgramInterface


class NetworkReceiver(ProgramInterface):
    def __init__(self, ip_address: str, port_number: int, buffer_size: Optional[int]):
        super().__init__()
        self.__ip_address = ip_address
        self.__port_number = port_number
        self.__buffer_size = buffer_size
    def get_program_name(self):
        return "Network Receiver"

    def get_command(self) -> str:
        command = fr"python {os.path.join('tasks/resources_consumers', 'network_receiver_task.py')} "
        command += f"-a {self.__ip_address} -p {self.__port_number}"
        if self.__buffer_size is not None:
            command += f" -s {self.__buffer_size}"
        return command