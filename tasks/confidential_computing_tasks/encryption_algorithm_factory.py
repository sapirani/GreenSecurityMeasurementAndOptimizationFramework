from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.paillier_security_algorithm import \
    PaillierSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL


class EncryptionAlgorithmFactory:
    @staticmethod
    def create_security_algorithm(encryption_algorithm: EncryptionType, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL) -> SecurityAlgorithm:
        if encryption_algorithm == EncryptionType.PaillierEncryption:
            return PaillierSecurityAlgorithm(min_key_val, max_key_val)
        else:
            raise ValueError("Unknown encryption algorithm")