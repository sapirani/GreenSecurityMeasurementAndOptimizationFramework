import os

from general_consts import MINUTE
from tasks.program_classes.abstract_program import ProgramInterface


class MemoryConsumer(ProgramInterface):
    def __init__(self, memory_chunk_size, consumption_speed, running_time):
        super().__init__()
        self.memory_chunk_size = memory_chunk_size
        self.consumption_speed = consumption_speed
        if running_time is None:
            self.running_time = 10 * MINUTE
        else:
            self.running_time = running_time

    def general_information_before_measurement(self, f):
        f.write(f"Memory Consumer - chunk size: {self.memory_chunk_size} bytes,"
                f" speed: {self.consumption_speed} bytes per second\n\n")

    def get_program_name(self):
        return "Memory Consumer"

    def get_command(self) -> str:
        return fr"python {os.path.join('tasks/DummyPrograms', 'DummyMemoryConsumer.py')} {self.memory_chunk_size} {self.consumption_speed} {self.running_time}"
