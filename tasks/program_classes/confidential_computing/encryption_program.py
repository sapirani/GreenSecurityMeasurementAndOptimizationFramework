import os
from abc import ABC, abstractmethod

from typing import Optional

from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.program_classes.abstract_program import ProgramInterface


CONFIDENTIAL_COMPUTING_TASKS_DIR = fr'tasks/confidential_computing_tasks'

class EncryptionProgram(ProgramInterface, ABC):
    def __init__(self, messages_file: str, results_file: str, security_algorithm: EncryptionType, key_file: str, encryption_task_path: str, min_key_value: Optional[int] = None, max_key_value: Optional[int] = None):
        super().__init__()
        self.__messages_file = messages_file
        self.__results_file = results_file
        self.__security_algorithm = security_algorithm.value
        self.__key_file = key_file
        self.__encryption_task_path = encryption_task_path
        self.__min_key_value = min_key_value
        self.__max_key_value = max_key_value

    @abstractmethod
    def get_program_name(self):
        pass

    def get_command(self) -> str:
        command = fr"python {os.path.join(CONFIDENTIAL_COMPUTING_TASKS_DIR, self.__encryption_task_path)} -m {self.__messages_file} -r {self.__results_file} -a {self.__security_algorithm} -k {self.__key_file}"
        if self.__min_key_value is not None and self.__max_key_value is not None:
            command += f" --min_key_val {self.__min_key_value} --max_key_val {self.__max_key_value}"
        return command
