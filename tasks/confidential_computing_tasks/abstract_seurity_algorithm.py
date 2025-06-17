from abc import ABC, abstractmethod

from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL


class SecurityAlgorithm(ABC):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        self._min_key_val = min_key_val
        self._max_key_val = max_key_val

    @abstractmethod
    def extract_key(self, key_file: str) -> KeyDetails:
        """ Initialize the public and private key """
        pass

    @abstractmethod
    def encrypt_message(self, msg: int) -> int:
        """ Encrypt the message """
        pass

    @abstractmethod
    def decrypt_message(self, msg: int) -> int:
        """ Decrypt the message """
        pass