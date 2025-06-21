import os
import pickle
from dataclasses import dataclass
from typing import Optional, Any

from Crypto.Cipher import AES, DES, Blowfish, ChaCha20, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
from Crypto.Util.Padding import pad, unpad

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


class PycryptodomeSymmetricAlgorithms:
    AES = "AES"
    DES = "DES"
    BLOWFISH = "BLOWFISH"
    CHACHA20 = "CHACHA20"
    ARC4 = "ARC4"

@dataclass
class AlgorithmDetails:
    name: str
    alg: Any
    block_size: int
    key_size: int
    is_stream: bool = False

class PycryptodomeKeyConsts:
    IV = "iv"
    NONCE = "nonce"
    COUNTER = "counter"
    KEY_HOLDER = "key"
    MODE = "mode"

class PycryptodomeSymmetricSecurityAlgorithm(SecurityAlgorithm[bytes]):
    __SUPPORTING_IV_MODES = ['CBC', 'CFB', 'OFB']
    __SUPPORTING_NONCE_MODES = ['CTR', 'EAX', 'GCM', PycryptodomeSymmetricAlgorithms.CHACHA20]
    __PADDING_MODES = ['ECB', 'CBC']

    __ALGORITHMS = {
        PycryptodomeSymmetricAlgorithms.AES: AlgorithmDetails(name="AES", alg=AES, block_size=AES.block_size, key_size=16),
        PycryptodomeSymmetricAlgorithms.DES: AlgorithmDetails(name="DES", alg=DES, block_size=DES.block_size, key_size=8),
        PycryptodomeSymmetricAlgorithms.BLOWFISH: AlgorithmDetails(name="Blowfish", alg=Blowfish, block_size=Blowfish.block_size, key_size=16),
        PycryptodomeSymmetricAlgorithms.CHACHA20: AlgorithmDetails(name="ChaCha20", alg=ChaCha20, block_size=1, key_size=32, is_stream=True),
        PycryptodomeSymmetricAlgorithms.ARC4: AlgorithmDetails(name="ARC4", alg=ARC4, block_size=1, key_size=16, is_stream=True),
        'DEFAULT': AlgorithmDetails(name="DEFAULT", alg=None, block_size=-1, key_size=-1)
    }

    __DEFAULT_KEY_STR = "DEFAULT"
    __MODEL_FILE = "encryption_model.bin"

    def __init__(self, algorithm: str, mode: Optional[str], min_key_val: int = PRIME_MIN_VAL,
                 max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.__algorithm = algorithm
        self.algorithm_name = algorithm.upper()
        self.mode_name = mode.upper()
        self.block_size = self._get_block_size()
        self.key = None
        self.nonce = None
        self.iv = None

    def _generate_key(self) -> bytes:
        alg_details = self.__ALGORITHMS.get(self.algorithm_name, self.__ALGORITHMS[self.__DEFAULT_KEY_STR])
        key_size = alg_details.key_size
        if key_size < 0:
            raise ValueError("Unsupported algorithm")

        return get_random_bytes(key_size)

    def _generate_iv(self) -> bytes:
        if self.mode_name in self.__SUPPORTING_IV_MODES:
            return get_random_bytes(self.block_size)
        return b''

    def _generate_nonce(self) -> bytes:
        if self.mode_name in self.__SUPPORTING_NONCE_MODES or self.algorithm_name == PycryptodomeSymmetricAlgorithms.CHACHA20:
            return get_random_bytes(8)  # Or 16 depending on mode requirements
        return b''

    def _get_block_size(self) -> int:
        alg_details = self.__ALGORITHMS.get(self.algorithm_name, self.__ALGORITHMS[self.__DEFAULT_KEY_STR])
        block_size = alg_details.block_size

        if block_size < 0:
            raise ValueError("Unsupported algorithm")
        return block_size

    def _get_cipher(self):
        alg_details = self.__ALGORITHMS.get(self.algorithm_name, self.__ALGORITHMS['DEFAULT'])

        if alg_details.is_stream:
            if self.algorithm_name == PycryptodomeSymmetricAlgorithms.CHACHA20:
                return alg_details.alg.new(key=self.key, nonce=self.nonce)
            elif self.algorithm_name == PycryptodomeSymmetricAlgorithms.ARC4:
                return alg_details.alg.new(key=self.key)
        else:
            mode = getattr(alg_details.alg, f'MODE_{self.mode_name}')
            kwargs = {PycryptodomeKeyConsts.KEY_HOLDER: self.key, PycryptodomeKeyConsts.MODE: mode}

            if self.mode_name in self.__SUPPORTING_IV_MODES:
                kwargs[PycryptodomeKeyConsts.IV] = self.iv
            if self.mode_name in self.__SUPPORTING_NONCE_MODES:
                kwargs[PycryptodomeKeyConsts.NONCE] = self.nonce
            if self.mode_name == 'CTR':
                kwargs[PycryptodomeKeyConsts.COUNTER] = Counter.new(self.block_size * 8)

            return alg_details.alg.new(**kwargs)

    def _get_algorithm_module(self):
        alg_details = self.__ALGORITHMS.get(self.algorithm_name, self.__ALGORITHMS[self.__DEFAULT_KEY_STR])
        alg_module = alg_details.alg
        if alg_module is None:
            raise ValueError("Unsupported algorithm")
        return alg_module

    def extract_key(self, key_file: str) -> KeyDetails:
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                data = pickle.load(f)
                self.key = data[PycryptodomeKeyConsts.KEY_HOLDER]
                self.iv = data.get(PycryptodomeKeyConsts.IV, b'')
                self.nonce = data.get(PycryptodomeKeyConsts.NONCE, b'')
        else:
            self.key = self._generate_key()
            self.iv = self._generate_iv()
            self.nonce = self._generate_nonce()
            with open(key_file, 'wb') as f:
                pickle.dump({
                    PycryptodomeKeyConsts.KEY_HOLDER: self.key,
                    PycryptodomeKeyConsts.IV: self.iv,
                    PycryptodomeKeyConsts.NONCE: self.nonce
                }, f)
        return KeyDetails(public_key={},
                          private_key={PycryptodomeKeyConsts.KEY_HOLDER: self.key,
                                       PycryptodomeKeyConsts.IV: self.iv,
                                       PycryptodomeKeyConsts.NONCE: self.nonce})

    def encrypt_message(self, msg: int) -> bytes:
        msg_bytes = str(msg).encode()
        cipher = self._get_cipher()
        if self.mode_name in self.__PADDING_MODES:
            return cipher.encrypt(pad(msg_bytes, self.block_size))
        else:
            return cipher.encrypt(msg_bytes)

    def decrypt_message(self, msg: bytes) -> int:
        cipher = self._get_cipher()
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
    pycr = PycryptodomeSymmetricSecurityAlgorithm("ARC4", "ECB")
    pycr.extract_key("pycr.key")
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
