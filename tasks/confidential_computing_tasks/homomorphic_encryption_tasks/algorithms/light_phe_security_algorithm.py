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
        self.__encryption_model = LightPHE(algorithm_name=algorithm)

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for LightPHE library.")
        return KeyDetails({}, {})

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


if __name__ == '__main__':
    algorithms = [
        "RSA",
        "ElGamal",
        "Exponential-ElGamal",
        "Paillier",
        "Damgard-Jurik",
        "Okamoto-Uchiyama",
        "Benaloh",
        "Naccache-Stern",
        "Goldwasser-Micali",
        "EllipticCurve-ElGamal"
    ]
    hom_type = "Paillier"
    a = 13
    b = 17
    cs = LightPHESecurityAlgorithm(hom_type)
    c1 = cs.encrypt_message(a)
    c2 = cs.encrypt_message(b)

    aa = cs.decrypt_message(c1)
    bb = cs.decrypt_message(c2)

    print("a == aa: ", a == aa)
    print("b == bb: ", b == bb)

    enc_sum = cs.add_messages(c1, c2)
    regular_sum_enc = cs.encrypt_message(a + b)

    dec_sum = cs.decrypt_message(enc_sum)
    regular_sum_dec = cs.decrypt_message(regular_sum_enc)

    print("d(e(a) + e(b)) == d(e(a+b)): ", dec_sum == regular_sum_dec)
    print("d(e(a) + e(b)) == a + b: ", dec_sum == (a + b))
    print("a + b == d(e(a+b)): ", (a + b) == regular_sum_dec)

    enc_mul = cs.multiply_messages(c1, c2)
    regular_mul_enc = cs.encrypt_message(a * b)

    dec_mul = cs.decrypt_message(enc_mul)
    regular_mul_dec = cs.decrypt_message(regular_mul_enc)

    print("d(e(a) * e(b)) == d(e(a*b)): ", dec_mul == regular_mul_dec)
    print("d(e(a) * e(b)) == a * b: ", dec_mul == (a * b))
    print("a * b == d(e(a*b)): ", (a * b) == regular_mul_dec)
