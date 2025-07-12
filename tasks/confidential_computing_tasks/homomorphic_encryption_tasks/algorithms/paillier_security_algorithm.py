import math
import random
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL


class PaillierKeyConsts:
    P_INDEX_IN_FILE = 0
    Q_INDEX_IN_FILE = 1

    G_PUBLIC_KEY = "g"
    N_PUBLIC_KEY = "n"

    P_PRIVATE_KEY = "p"
    Q_PRIVATE_KEY = "q"
    LMBDA_PRIVATE_KEY = "lmbda"
    MU_PRIVATE_KEY = "mu"

    NUM_OF_KEY_PARTS = 2


class PaillierSecurityAlgorithm(HomomorphicSecurityAlgorithm[int]):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.p = None
        self.q = None
        self.n = None
        self.phi = None
        self.g = None
        self.lmbda = None
        self.mu = None

    def _generate_and_save_key(self, key_file) -> KeyDetails:
        p, q = self._extract_random_prime_p_and_q(key_file, should_generate=True,
                                                  num_of_key_parts=PaillierKeyConsts.NUM_OF_KEY_PARTS)
        return self.__create_full_key(p, q)

    def _load_key(self, key_file) -> KeyDetails:
        p, q = self._extract_random_prime_p_and_q(key_file, should_generate=False,
                                                  num_of_key_parts=PaillierKeyConsts.NUM_OF_KEY_PARTS)
        return self.__create_full_key(p, q)

    def __create_full_key(self, p: int, q: int) -> KeyDetails:
        """ Initialize the public and private key """
        if self.p is not None and self.q is not None:
            raise RuntimeError("Key is already initialized")

        self.p, self.q = p, q
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

    def add_messages(self, c1: int, c2: int) -> int:
        return (c1 * c2) % (self.n * self.n)

    def multiply_messages(self, c1: int, c2: int) -> int:
        raise NotImplementedError("Paillier Homomorphic Encryption does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: int, scalar: int) -> int:
        return c * scalar
