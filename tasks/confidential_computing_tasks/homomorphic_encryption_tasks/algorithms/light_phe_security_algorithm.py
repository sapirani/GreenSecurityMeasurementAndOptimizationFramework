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
    def __init__(self, algorithm: str, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__algorithm = algorithm
        self.__encryption_model = None


    def _generate_and_save_key(self, key_file) -> KeyDetails:
        if self.__encryption_model is not None:
            raise RuntimeError("Encryption model already initialized for LightPHE library.")
        try:
            self.__encryption_model = LightPHE(algorithm_name=self.__algorithm)
            with open(key_file, "wb") as f:
                pickle.dump(self.__encryption_model, f)
        except Exception as e:
            raise RuntimeError("Error occurred when saving lightPhe model.")

        return KeyDetails(public_key={}, private_key={"model": self.__encryption_model})

    def _load_key(self, key_file) -> KeyDetails:
        try:
            with open(key_file, "rb") as f:
                self.__encryption_model = pickle.load(f)
        except Exception as e:
            raise RuntimeError("Error occurred when loading lightPhe model.")
        return KeyDetails(public_key={}, private_key={"model": self.__encryption_model})


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
