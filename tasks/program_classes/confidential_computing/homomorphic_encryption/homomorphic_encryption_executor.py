from typing import Optional

from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.program_classes.confidential_computing.encryption_program import EncryptionProgram

HOMOMORPHIC_ENCRYPTION_PIPELINE = r'homomorphic_encryption_tasks/homomorphic_pipeline.py'


class HomomorphicEncryptionExecutor(EncryptionProgram):
    def __init__(self, messages_file: str, security_algorithm: EncryptionType, key_file: str,
                 min_key_value: Optional[int] = None, max_key_value: Optional[int] = None):
        super().__init__(messages_file, security_algorithm, key_file, HOMOMORPHIC_ENCRYPTION_PIPELINE, min_key_value,
                         max_key_value)

    def get_program_name(self):
        return "Homomorphic Encryption Pipeline"
