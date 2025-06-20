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

    NUM_OF_KEY_PARTS = 2


class RSASecurityAlgorithm(HomomorphicSecurityAlgorithm[int]):
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

        self.p, self.q = self._extract_random_prime_p_and_q(key_file, RSAKeyConsts.NUM_OF_KEY_PARTS)
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

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[int]) -> list[int]:
        return encrypted_messages

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[int]) -> list[int]:
        return encrypted_messages

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
        return (c1 * c2) % self.n

    def scalar_and_message_multiplication(self, c: int, scalar: int) -> int:
        raise NotImplementedError("RSA Homomorphic Encryption does not support multiplying message with scalar.")
