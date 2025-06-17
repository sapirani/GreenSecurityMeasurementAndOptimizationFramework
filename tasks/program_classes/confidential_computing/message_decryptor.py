from typing import Optional

from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.program_classes.confidential_computing.encryption_program import EncryptionProgram

MESSAGE_DECRYPTION_TASK = r'decrypt_messages.py'


class MessageDecryptor(EncryptionProgram):
    def __init__(self, messages_file: str, security_algorithm: EncryptionType, key_file: str,
                 min_key_value: Optional[int] = None, max_key_value: Optional[int] = None):
        super().__init__(messages_file, security_algorithm, key_file, MESSAGE_DECRYPTION_TASK, min_key_value,
                         max_key_value)

    def get_program_name(self):
        return "Messages Decryptor"
