import pickle

from lightphe import LightPHE, Ciphertext

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


class LightPHEAlgorithms:
    RSA = "RSA"
    ELGAMAL = "ElGamal"
    EXPONENTIAL_ELGAMAL = "Exponential-ElGamal"
    PAILLIER = "Paillier"
    DAMGARD_JURIK = "Damgard-Jurik"
    OKAMOTO_UCHIYAMA = "Okamoto-Uchiyama"
    BENALOH = "Benaloh"
    NACCACHE_STERN = "Naccache-Stern"
    GOLDWASSER_MICALI = "Goldwasser-Micali"
    ELLIPTICCURVE_ELGAMAL = "EllipticCurve-ElGamal"


class LightPHESecurityAlgorithm(HomomorphicSecurityAlgorithm[Ciphertext]):
    __MODEL_FILE = "encryption_model.bin"

    def __init__(self, algorithm: str, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__algorithm = algorithm
        self.__encryption_model = LightPHE(algorithm_name=algorithm)

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for LightPHE library.")
        return KeyDetails({}, {})

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[Ciphertext]) -> list[Ciphertext]:
        try:
            with open(self.__MODEL_FILE, "wb") as f:
                pickle.dump(self.__encryption_model, f)
        except Exception as e:
            raise RuntimeError("Error occurred when saving lightPhe model.")
        return encrypted_messages

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[Ciphertext]) -> list[Ciphertext]:
        try:
            with open(self.__MODEL_FILE, 'rb') as messages_file:
                self.__encryption_model = pickle.load(messages_file)
        except FileNotFoundError:
            print("Something went wrong with loading the encrypted messages")

        return encrypted_messages

    def encrypt_message(self, msg: int) -> Ciphertext:
        """ Encrypt the message """
        return self.__encryption_model.encrypt(msg)

    def decrypt_message(self, msg: Ciphertext) -> int:
        """ Decrypt the message """
        return self.__encryption_model.decrypt(msg)

    def add_messages(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        try:
            return c1 + c2
        except Exception as e:
            raise NotImplementedError(f"LightPHE with algorithm {self.__algorithm} does not support adding messages.")

    def multiply_messages(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        try:
            return c1 * c2
        except Exception as e:
            raise NotImplementedError(f"LightPHE with algorithm {self.__algorithm} does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: Ciphertext, scalar: int) -> Ciphertext:
        try:
            return scalar * c
        except Exception as e:
            raise NotImplementedError(f"LightPHE with algorithm {self.__algorithm} does not support multiplying message with scalar.")
