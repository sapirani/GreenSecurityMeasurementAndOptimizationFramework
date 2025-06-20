import os
import pickle
from typing import Optional

from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

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
    __SUPPORTING_IV_MODES = ['CBC', 'CFB', 'OFB']
    __SUPPORTING_NONCE_MODES = ['CTR', 'EAX', 'GCM']
    __PADDING_MODES = ['ECB', 'CBC']

    __ALGORITHMS = {
        'AES': AES,
        'DES': DES,
        'BLOWFISH': Blowfish
    }

    __BLOCK_SIZES = {
            'AES': AES.block_size,
            'DES': DES.block_size,
            'BLOWFISH': Blowfish.block_size,
            "default": 16
        }

    __KEY_SIZES = {
            'AES': 16,
            'DES': 8,
            'BLOWFISH': 16,
            "default": 16
        }

    __DEFAULT_KEY_STR = "default"
    __MODEL_FILE = "encryption_model.bin"

    def __init__(self, algorithm: str, mode: Optional[str], min_key_val: int = PRIME_MIN_VAL,
                 max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__algorithm = algorithm
        self.__mode = mode
        self.algorithm_name = algorithm.upper()
        self.mode_name = mode.upper()
        self.block_size = self._get_block_size()
        self.nonce = self._generate_nonce()
        self.key = self._generate_key()
        self.iv = self._generate_iv()

    def _generate_key(self) -> bytes:
        key_size = self.__KEY_SIZES.get(self.algorithm_name, -1)
        if key_size < 0:
            raise ValueError("Unsupported algorithm")

        return get_random_bytes(key_size)

    def _generate_iv(self) -> bytes:
        if self.mode_name in self.__SUPPORTING_IV_MODES:
            return get_random_bytes(self.block_size)
        return b''

    def _generate_nonce(self) -> bytes:
        if self.mode_name in self.__SUPPORTING_NONCE_MODES:
            return get_random_bytes(8)  # Or 16 depending on mode requirements
        return b''

    def _get_block_size(self) -> int:
        return self.__BLOCK_SIZES.get(self.algorithm_name, self.__BLOCK_SIZES[self.__DEFAULT_KEY_STR])

    def _get_cipher(self, encrypting: bool):
        mode = getattr(self._get_algorithm_module(), f'MODE_{self.mode_name}')
        kwargs = {'key': self.key}

        if self.mode_name in self.__SUPPORTING_IV_MODES:
            kwargs['iv'] = self.iv

        if self.mode_name in self.__SUPPORTING_NONCE_MODES:
            kwargs["nonce"] = self.nonce

        if self.mode_name == 'CTR':
            from Crypto.Util import Counter
            kwargs['counter'] = Counter.new(self.block_size * 8)

        if encrypting:
            return self._get_algorithm_module().new(**kwargs, mode=mode)
        else:
            return self._get_algorithm_module().new(**kwargs, mode=mode)

    def _get_algorithm_module(self):
        return self.__ALGORITHMS.get(self.algorithm_name, AES)

    def extract_key(self, key_file: str) -> KeyDetails:
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                data = pickle.load(f)
                self.key = data['key']
                self.iv = data.get('iv', b'')
        else:
            with open(key_file, 'wb') as f:
                pickle.dump({'key': self.key, 'iv': self.iv}, f)
        return KeyDetails(public_key={}, private_key={'key': self.key, 'iv': self.iv})

    def encrypt_message(self, msg: int) -> bytes:
        msg_bytes = str(msg).encode()
        cipher = self._get_cipher(encrypting=True)
        if self.mode_name in self.__PADDING_MODES:
            return cipher.encrypt(pad(msg_bytes, self.block_size))
        else:
            return cipher.encrypt(msg_bytes)

    def decrypt_message(self, msg: bytes) -> int:
        cipher = self._get_cipher(encrypting=False)
        if self.mode_name in self.__PADDING_MODES:
            plaintext = unpad(cipher.decrypt(msg), self.block_size)
        else:
            plaintext = cipher.decrypt(msg)
        return int(plaintext.decode())

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        return encrypted_messages

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[bytes]) -> list[bytes]:
        return encrypted_messages


if __name__ == "__main__":
    pycr = PycryptodomeSecurityAlgorithm("AES", "ECB")
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
