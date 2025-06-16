from abc import ABC, abstractmethod

from tasks.confidential_computing_tasks.key_details import KeyDetails


class SecurityAlgorithm(ABC):

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