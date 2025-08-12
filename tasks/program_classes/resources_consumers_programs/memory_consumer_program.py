import os

from utils.general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class MemoryConsumer(ProgramInterface):
    def __init__(self, memory_chunk_size: int, consumption_speed: float):
        super().__init__()
        self.memory_chunk_size = memory_chunk_size
        self.consumption_speed = consumption_speed

    def general_information_before_measurement(self, f):
        f.write(f"Memory Consumer - chunk size: {self.memory_chunk_size} bytes,"
                f" speed: {self.consumption_speed} bytes per second\n\n")

    def get_program_name(self):
        return "Memory Consumer"

    def get_command(self) -> str:
        return f"python {os.path.join('tasks/resources_consumers', 'memory_consumer_task.py')} -c {self.memory_chunk_size} -s {self.consumption_speed}"
