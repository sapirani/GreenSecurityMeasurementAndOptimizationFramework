import os

from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.program_classes.abstract_program import ProgramInterface

CONFIDENTIAL_COMPUTING_TASKS_DIR = fr'tasks/confidential_computing_tasks'
MESSAGE_DECRYPTION_TASK = r'decrypt_messages.py'

class MessageDecryptor(ProgramInterface):
    def __init__(self, messages_file: str, security_algorithm: EncryptionType, key_file: str):
        super().__init__()
        self.__messages_file = messages_file
        self.__security_algorithm = security_algorithm.value
        self.__key_file = key_file

    def get_program_name(self):
        return "Messages Decryptor"

    def get_command(self) -> str:
        return fr"python {os.path.join(CONFIDENTIAL_COMPUTING_TASKS_DIR, MESSAGE_DECRYPTION_TASK)} -m {self.__messages_file} -a {self.__security_algorithm} -k {self.__key_file}"
