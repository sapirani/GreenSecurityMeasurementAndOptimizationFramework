from typing import Optional

from tasks.confidential_computing_tasks.encryption_type import EncryptionMode, EncryptionType
from tasks.program_classes.confidential_computing.encryption_program import EncryptionProgram

MESSAGE_MULTIPLICATION_TASK = r'message_multiplier.py'

class MessageMultiplier(EncryptionProgram):
    def __init__(self, messages_file: str, results_file: str, security_algorithm: EncryptionType,
                 block_mode: Optional[EncryptionMode], key_file: str, min_key_value: Optional[int] = None, max_key_value: Optional[int] = None):
        super().__init__(messages_file, results_file, security_algorithm, block_mode, key_file, MESSAGE_MULTIPLICATION_TASK,
                         min_key_value, max_key_value)

    def get_program_name(self):
        return "Messages Multiplication (Encryption)"