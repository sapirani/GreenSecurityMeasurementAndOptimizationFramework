import math
import random
from tasks.confidential_computing_tasks.basic_utils import generate_random_prime
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL


class RSAKeyConsts:
    P_INDEX_IN_FILE = 0
    Q_INDEX_IN_FILE = 1

    E_PUBLIC_KEY = "e"
    N_PUBLIC_KEY = "n"

    P_PRIVATE_KEY = "p"
    Q_PRIVATE_KEY = "q"
    D_PRIVATE_KEY = "d"


class RSASecurityAlgorithm(HomomorphicSecurityAlgorithm):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.p = None
        self.q = None
        self.n = None
        self.phi = None
        self.e = None
        self.d = None

    def extract_key(self, key_file: str) -> KeyDetails:
        """ Initialize the public and private key """
        if self.p is not None and self.q is not None:
            raise RuntimeError("Key is already initialized")

        try:
            with open(key_file, "r") as f:
                key_lines = f.readlines()
        except FileNotFoundError:
            key_lines = []

        if len(key_lines) != 2:
            self.p, self.q = self.__generate_initial_primes(key_file)
            print("Generated p, q randomly.")
        else:
            self.p = int(key_lines[RSAKeyConsts.P_INDEX_IN_FILE].strip())
            self.q = int(key_lines[RSAKeyConsts.Q_INDEX_IN_FILE].strip())
            print(f"Extracted p, q from {key_file}.")

        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        # Simplified variant parameters
        self.e = self.phi - 1
        self.d = pow(self.e, -1, self.phi)

        public_key = {RSAKeyConsts.E_PUBLIC_KEY: self.e,
                      RSAKeyConsts.N_PUBLIC_KEY: self.n}
        private_key = {RSAKeyConsts.P_PRIVATE_KEY: self.p,
                       RSAKeyConsts.Q_PRIVATE_KEY: self.q,
                       RSAKeyConsts.D_PRIVATE_KEY: self.d}
        return KeyDetails(public_key=public_key, private_key=private_key)

    def __generate_initial_primes(self, key_file: str) -> tuple[int, int]:
        min_prime_number = math.isqrt(self._min_key_val)
        max_prime_number = math.isqrt(self._max_key_val)
        self.p = generate_random_prime(min_prime_number, max_prime_number)
        self.q = generate_random_prime(min_prime_number, max_prime_number)

        while self.p == self.q:
            self.q = generate_random_prime(min_prime_number, max_prime_number)

        self.__save_key(key_file)
        return self.p, self.q

    def __save_key(self, key_file: str):
        with open(key_file, "w") as f:
            f.write(f"{self.p}\n{self.q}")

    def encrypt_message(self, msg: int) -> int:
        """ Encrypt the message """
        if self.n is None:
            raise RuntimeError("Key is not initialized. Call extract_key first.")

        if msg >= self.n:
            raise ValueError(f"Message {msg} is too large for RSA modulus n={self.n}")

        encrypted_message = pow(msg, self.e, self.n)
        return encrypted_message

    def decrypt_message(self, msg: int) -> int:
        """ Decrypt the message """
        if self.n is None:
            raise RuntimeError("Key is not initialized. Call extract_key first.")

        decrypted_message = pow(msg, self.d, self.n)
        return decrypted_message

    def add_messages(self, c1: int, c2: int) -> int:
        raise NotImplementedError("RSA Homomorphic Encryption does not support addition of messages.")

    def multiply_messages(self, c1: int, c2: int) -> int:
        return c1 * c2
