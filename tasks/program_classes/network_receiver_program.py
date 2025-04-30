import os

from tasks.program_classes.abstract_program import ProgramInterface


class NetworkReceiver(ProgramInterface):
    def get_program_name(self):
        return "Network Receiver"

    def get_command(self) -> str:
        return fr"python {os.path.join('tasks/DummyPrograms', 'DummyNetworkReceiver.py')}"
