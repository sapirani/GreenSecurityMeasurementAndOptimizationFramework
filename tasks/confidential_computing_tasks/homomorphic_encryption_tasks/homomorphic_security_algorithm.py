import math
from abc import ABC, abstractmethod
from typing import Optional, Callable

from typing_extensions import override

from tasks.confidential_computing_tasks.abstract_security_algorithm import SecurityAlgorithm, T
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
    def calc_encrypted_sum(self, messages: list[int], start_total: Optional[T] = None,
                           checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        return self.__calc_encrypted_operation(messages, is_addition=True, start_total=start_total, checkpoint_callback=checkpoint_callback)

    @override
    def calc_encrypted_multiplication(self, messages: list[int], start_total: Optional[T] = None,
                                      checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        return self.__calc_encrypted_operation(messages, is_addition=False, start_total=start_total, checkpoint_callback=checkpoint_callback)

    def __calc_encrypted_operation(self, messages: list[int], *, is_addition: bool, start_total: Optional[T] = None,
                                   checkpoint_callback: Optional[Callable[[T, T], None]] = None) -> T:
        total = start_total if start_total is not None else self.encrypt_message(messages[0])
        start_idx = 0 if start_total is not None else 1

        for i in range(start_idx, len(messages)):
            encrypted = self.encrypt_message(messages[i])
            total = self.add_messages(total, encrypted) if is_addition else self.multiply_messages(total, encrypted)
            if checkpoint_callback:
                checkpoint_callback(encrypted, total)

        return total

    @abstractmethod
    def _generate_and_save_key(self, key_file) -> KeyDetails:
        pass

    @abstractmethod
    def _load_key(self, key_file) -> KeyDetails:
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

    def _extract_random_prime_p_and_q(self, key_file: str, should_generate: bool,
                                      num_of_key_parts: int = KeyConsts.DEFAULT_KEY_PARTS) -> \
            tuple[int, int]:

        if should_generate:
            print("Generated p, q randomly.")
            return self.__generate_initial_primes(key_file)
        else:
            try:
                with open(key_file, "r") as f:
                    key_lines = f.readlines()
            except FileNotFoundError:
                raise Exception("Key file not found.")

            if len(key_lines) == num_of_key_parts:
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
