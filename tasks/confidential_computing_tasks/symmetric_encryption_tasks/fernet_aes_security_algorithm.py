from typing import Literal

from cryptography.fernet import Fernet

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


class FernetAESSecurityAlgorithm(SecurityAlgorithm[bytes]):
    __KEY_STR = "key"
    __NUM_OF_BYTES = 2
    __ORDER: Literal["little", "big"] = "big"

    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__key = None
        self.__encryption_fernet = None

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        return encrypted_messages

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        return encrypted_messages

    def _generate_and_save_key(self, key_file) -> KeyDetails:
        if self.__key is not None or self.__encryption_fernet is not None:
            raise Exception("Key is already generated for Fernet AES encryption.")

        print("Generated new fernet key randomly.")
        self.__key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(self.__key)

        self.__encryption_fernet = Fernet(self.__key)
        return KeyDetails(public_key={}, private_key={self.__KEY_STR: self.__key})

    def _load_key(self, key_file) -> KeyDetails:
        try:
            with open(key_file, "r") as f:
                key_content = f.read().strip()
        except FileNotFoundError:
            key_content = ""

        if key_content:
            self.__key = key_content
            print(f"Extracted fernet key from {key_file}.")
        else:
            raise Exception("Key file is not found or empty. Call extract_key() with should_generate=True.")

        self.__encryption_fernet = Fernet(self.__key)
        return KeyDetails(public_key={}, private_key={self.__KEY_STR: self.__key})

    def encrypt_message(self, msg: int) -> bytes:
        """ Encrypt the message """
        return self.__encryption_fernet.encrypt(msg.to_bytes(self.__NUM_OF_BYTES, self.__ORDER))

    def decrypt_message(self, msg: bytes) -> int:
        """ Decrypt the message """
        decrypted_msg = self.__encryption_fernet.decrypt(msg)
        return int.from_bytes(decrypted_msg, self.__ORDER)
