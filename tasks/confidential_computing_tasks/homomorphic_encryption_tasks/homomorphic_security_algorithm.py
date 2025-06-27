import math
from abc import ABC, abstractmethod

from typing_extensions import override

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm, T
from tasks.confidential_computing_tasks.utils.basic_utils import generate_random_prime
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails

class KeyConsts:
    P_INDEX_IN_FILE = 0
    Q_INDEX_IN_FILE = 1

    DEFAULT_KEY_PARTS = 2

class HomomorphicSecurityAlgorithm(SecurityAlgorithm[T], ABC):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)

    @override
    def calc_encrypted_sum(self, messages: list[int]) -> T:
        return self.__calc_encrypted_operation(messages, True)

    @override
    def calc_encrypted_multiplication(self, messages: list[int]) -> T:
        return self.__calc_encrypted_operation(messages, False)

    def __calc_encrypted_operation(self, messages: list[int], is_addition: bool) -> T:
        encrypted_messages = [self.encrypt_message(msg) for msg in messages]
        total_encrypted_result = encrypted_messages[0]
        for enc_message in encrypted_messages[1:]:
            if is_addition:
                total_encrypted_result = self.add_messages(total_encrypted_result, enc_message)
            else:
                total_encrypted_result = self.multiply_messages(total_encrypted_result, enc_message)
        return total_encrypted_result

    @abstractmethod
    def extract_key(self, key_file: str) -> KeyDetails:
        """ Initialize the public and private key """
        pass

    @abstractmethod
    def encrypt_message(self, msg: int) -> T:
        """ Encrypt the message """
        pass

    @abstractmethod
    def decrypt_message(self, msg: T) -> int:
        """ Decrypt the message """
        pass

    @abstractmethod
    def add_messages(self, c1: T, c2: T) -> T:
        pass

    @abstractmethod
    def multiply_messages(self, c1: T, c2: T) -> T:
        pass

    @abstractmethod
    def scalar_and_message_multiplication(self, c: T, scalar: int) -> T:
        pass

    def _extract_random_prime_p_and_q(self, key_file: str, num_of_key_parts: int = KeyConsts.DEFAULT_KEY_PARTS) -> \
            tuple[int, int]:
        try:
            with open(key_file, "r") as f:
                key_lines = f.readlines()
        except FileNotFoundError:
            key_lines = []

        if len(key_lines) != num_of_key_parts:
            print("Generated p, q randomly.")
            return self.__generate_initial_primes(key_file)
        else:
            p = int(key_lines[KeyConsts.P_INDEX_IN_FILE].strip())
            q = int(key_lines[KeyConsts.Q_INDEX_IN_FILE].strip())
            print(f"Extracted p, q from {key_file}.")
            return p, q

    def __generate_initial_primes(self, key_file: str) -> tuple[int, int]:
        min_prime_number = math.isqrt(self._min_key_val)
        max_prime_number = math.isqrt(self._max_key_val)
        p = generate_random_prime(min_prime_number, max_prime_number)
        q = generate_random_prime(min_prime_number, max_prime_number)

        while p == q:
            q = generate_random_prime(min_prime_number, max_prime_number)

        self.__save_p_q_key(key_file, p, q)
        return p, q

    def __save_p_q_key(self, key_file: str, p: int, q: int):
        with open(key_file, "w") as f:
            f.write(f"{p}\n{q}")