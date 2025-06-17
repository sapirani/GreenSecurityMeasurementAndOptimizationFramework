import math
import random
from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.basic_utils import generate_random_prime
from tasks.confidential_computing_tasks.key_details import KeyDetails


PRIME_MIN_VAL = 2 ** 1023 - 1
PRIME_MAX_VAL = 2 ** 1024 - 1

class PaillierKeyConsts:
    P_INDEX_IN_FILE = 0
    Q_INDEX_IN_FILE = 1

    G_PUBLIC_KEY = "g"
    N_PUBLIC_KEY = "n"

    P_PRIVATE_KEY = "p"
    Q_PRIVATE_KEY = "q"
    LMBDA_PRIVATE_KEY = "lmbda"
    MU_PRIVATE_KEY = "mu"


class PaillierSecurityAlgorithm(SecurityAlgorithm):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        self.p = None
        self.q = None
        self.n = None
        self.phi = None
        self.g = None
        self.lmbda = None
        self.mu = None
        self.__min_key_val = min_key_val
        self.__max_key_val = max_key_val

    def extract_key(self, key_file: str) -> KeyDetails:
        """ Initialize the public and private key """
        if self.p is not None and self.q is not None:
            raise RuntimeError("Key is already initialized")

        try:
            with open(key_file, "r") as key_file:
                key_lines = key_file.readlines()
        except FileNotFoundError:
            key_lines = []

        if len(key_lines) != 2:
            self.p = generate_random_prime(self.__min_key_val, self.__max_key_val)
            self.q = generate_random_prime(self.__min_key_val, self.__max_key_val)
            self.__save_key(key_file)
            print("Generated p, q randomly.")
        else:
            self.p = int(key_lines[PaillierKeyConsts.P_INDEX_IN_FILE].strip())
            self.q = int(key_lines[PaillierKeyConsts.Q_INDEX_IN_FILE].strip())
            print(f"Extracted p, q from {key_file}.")

        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        # Simplified variant parameters
        self.g = self.n + 1
        self.lmbda = self.phi
        self.mu = pow(self.lmbda, -1, self.n)

        public_key = {PaillierKeyConsts.G_PUBLIC_KEY: self.g,
                      PaillierKeyConsts.N_PUBLIC_KEY: self.n}
        private_key = {PaillierKeyConsts.P_PRIVATE_KEY: self.p,
                       PaillierKeyConsts.Q_PRIVATE_KEY: self.q,
                       PaillierKeyConsts.LMBDA_PRIVATE_KEY: self.lmbda,
                       PaillierKeyConsts.MU_PRIVATE_KEY: self.mu}
        return KeyDetails(public_key=public_key, private_key=private_key)

    def __save_key(self, key_file: str):
        with open(key_file, "w") as key_file:
            key_file.write(f"{self.p}\n{self.q}")

    def encrypt_message(self, msg: int) -> int:
        """ Encrypt the message """
        if self.n is None:
            raise RuntimeError("Key is not initialized. Call extract_key first.")

        n_pow = self.n * self.n
        r = self.__get_r_for_encryption(self.n)
        encrypted_message = (pow(self.g, msg, n_pow) * pow(r, self.n, n_pow)) % n_pow
        return encrypted_message

    def __get_r_for_encryption(self, n) -> int:
        """Return random r for encryption"""
        r = random.randint(1, n - 1)
        while math.gcd(r, n) != 1:
            r = random.randint(1, n - 1)
        return r

    def decrypt_message(self, msg: int) -> int:
        """ Decrypt the message """
        if self.n is None:
            raise RuntimeError("Key is not initialized. Call extract_key first.")

        cl = pow(msg, self.lmbda, self.n * self.n)
        l = int(cl - 1) / int(self.n)
        decrypted_msg = int((l * self.mu) % self.n)

        # Handle negative values (convert from modular arithmetic)
        if decrypted_msg > self.n // 2:
            decrypted_msg = decrypted_msg - self.n

        return decrypted_msg
