from typing import Literal

from cryptography.fernet import Fernet

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


class FernerAESSecurityAlgorithm(SecurityAlgorithm[bytes]):
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

    def extract_key(self, key_file: str) -> KeyDetails:
        """ Initialize the public and private key """
        try:
            with open(key_file, "r") as f:
                key_lines = f.readlines()
        except FileNotFoundError:
            key_lines = []

        if len(key_lines) != 1:
            print("Generated new fernet key randomly.")
            self.__key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.__key)
        else:
            self.__key = key_lines[0].strip()
            print(f"Extracted fernet key from {key_file}.")

        self.__encryption_fernet = Fernet(self.__key)
        return KeyDetails(public_key={}, private_key={self.__KEY_STR: self.__key})

    def encrypt_message(self, msg: int) -> bytes:
        """ Encrypt the message """
        return self.__encryption_fernet.encrypt(msg.to_bytes(self.__NUM_OF_BYTES, self.__ORDER))

    def decrypt_message(self, msg: bytes) -> int:
        """ Decrypt the message """
        decrypted_msg = self.__encryption_fernet.decrypt(msg)
        return int.from_bytes(decrypted_msg, self.__ORDER)

# if __name__ == "__main__":
#     ferner = FernerAESSecurityAlgorithm()
#     ferner.extract_key("fernet.key")
#     m1 = 56
#     m2 = 84
#
#     c1 = ferner.encrypt_message(m1)
#     c2 = ferner.encrypt_message(m2)
#
#     sum = m2 + m1
#     sum_c = ferner.encrypt_message(sum)
#     sum_c_dec = ferner.decrypt_message(sum_c)
#
#     m11 = ferner.decrypt_message(c1)
#     m22 = ferner.decrypt_message(c2)
#     sum_dec = m11 + m22
#
#     print(f"m11: {m11}")
#     print(f"m22: {m22}")
#     print(f"sum_dec == sum_c_dec: {sum_dec == sum_c_dec}")