from typing import Optional

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


class PycryptodomeSymmetricAlgorithms:
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


class PycryptodomeSecurityAlgorithm(SecurityAlgorithm[bytes]):
    __MODEL_FILE = "encryption_model.bin"

    def __init__(self, algorithm: str, mode: Optional[str], min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__key = get_random_bytes(16)
        self.__algorithm = algorithm
        self.__mode = mode
        self.__encryption_model = self.__pycryptodome_algorithm_factory(algorithm, mode, None)
        self.__decryption_model = self.__pycryptodome_algorithm_factory(algorithm, mode, self.__encryption_model.nonce)

    def __pycryptodome_algorithm_factory(self, algorithm: str, mode: str, nonce: Optional[bytes]):
        extra = {}
        if nonce is not None:
            extra["nonce"] = nonce

        if algorithm == "AES" and mode == "cbc":
            return AES.new(self.__key, AES.MODE_CTR, **extra)
        raise ValueError("Unknown encryption algorithm")

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for LightPHE library.")
        return KeyDetails({}, {})

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        try:
            # with open(self.__MODEL_FILE, "wb") as f:
            #     pickle.dump(self.__encryption_model, f)
            return encrypted_messages
        except Exception as e:
            raise RuntimeError("Error occurred when saving lightPhe model.")
        return encrypted_messages

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        try:
            # with open(self.__MODEL_FILE, 'rb') as messages_file:
            #     self.__encryption_model = pickle.load(messages_file)
            return encrypted_messages
        except FileNotFoundError:
            print("Something went wrong with loading the encrypted messages")

        return encrypted_messages

    def encrypt_message(self, msg: int) -> bytes:
        """ Encrypt the message """
        return self.__encryption_model.encrypt(msg.to_bytes(2, "big"))

    def decrypt_message(self, msg: bytes) -> int:
        """ Decrypt the message """
        return int.from_bytes(self.__decryption_model.decrypt(msg), byteorder="big")

if __name__ == "__main__":
    pycr = PycryptodomeSecurityAlgorithm("AES", "cbc")
    m1 = 56
    m2 = 83

    c1 = pycr.encrypt_message(m1)
    c2 = pycr.encrypt_message(m2)

    m11 = pycr.decrypt_message(c1)
    m12 = pycr.decrypt_message(c2)

    sum_reg = m1 + m2
    sum_enc = pycr.encrypt_message(sum_reg)
    sec_sum_reg = pycr.decrypt_message(sum_enc)

    sum_new = m11 + m12
    sum_enc_new = pycr.encrypt_message(sum_new)
    sum_dec_new = pycr.decrypt_message(sum_enc_new)

    print(f"M11: {m11}")
    print(f"M12: {m12}")
    print(f"Sum: {sum_reg}")
    print(f"Sum_new: {sum_dec_new}")