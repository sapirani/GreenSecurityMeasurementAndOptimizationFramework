import os
from typing import Optional

from utils.general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class NetworkSender(ProgramInterface):
    def __init__(self, time_interval: float, running_time: int, ip_address: str, port_number: int, packet_size: Optional[int]):
        super().__init__()
        self.__ip_address = ip_address
        self.__port_number = port_number
        self.__packet_size = packet_size
        self.__time_interval = time_interval
        if running_time is None:
            self.__running_time = 10 * MINUTE
        else:
            self.__running_time = running_time

    def general_information_before_measurement(self, f):
        f.write(f"Network Sender - time interval: {self.__time_interval}\n\n")

    def get_program_name(self):
        return "Network Sender"

    def get_command(self) -> str:
        command = fr"python {os.path.join('tasks/resources_consumers', 'network_sender_task.py')} "
        command += f"-a {self.__ip_address} -p {self.__port_number} -i {self.__time_interval} -t {self.__running_time}"
        if self.__packet_size is not None:
            command += f" -s {self.__packet_size}"
        return command
