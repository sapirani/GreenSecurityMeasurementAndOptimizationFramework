from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.paillier_security_algorithm import \
    PaillierSecurityAlgorithm


class EncryptionAlgorithmFactory:
    @staticmethod
    def create_security_algorithm(encryption_algorithm: EncryptionType) -> SecurityAlgorithm:
        if encryption_algorithm == EncryptionType.PaillierEncryption:
            return PaillierSecurityAlgorithm()
        else:
            raise ValueError("Unknown encryption algorithm")