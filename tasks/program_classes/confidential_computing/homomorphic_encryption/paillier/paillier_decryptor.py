import os

from tasks.program_classes.abstract_program import ProgramInterface

PAILLIER_TASKS_DIR = fr'tasks/confidential_computing_tasks/homomorphic_encryption_tasks/paillier_encryption'

class PaillierDecryptor(ProgramInterface):
    def __init__(self, p: int, q: int, msg_to_decrypt: int):
        super().__init__()
        self.__p = p
        self.__q = q
        self.__msg_to_decrypt = msg_to_decrypt

    def get_program_name(self):
        return "Paillier Decryptor"

    def get_command(self) -> str:
        return fr"python {os.path.join(PAILLIER_TASKS_DIR, 'decrypt_large_number.py')} {self.__msg_to_decrypt} {self.__p} {self.__q}"
