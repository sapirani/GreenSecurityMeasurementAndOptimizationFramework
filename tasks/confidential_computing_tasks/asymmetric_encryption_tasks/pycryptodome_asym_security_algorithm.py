from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import pickle
import os

from tasks.confidential_computing_tasks.abstract_security_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MAX_VAL, PRIME_MIN_VAL


class PycryptodomeAsymmetricAlgorithms:
    RSA = "RSA"


class PycryptodomeKeyConsts:
    KEY_HOLDER = "key"

    PRIVATE_KEY = "private_key"
    PUBLIC_KEY = "public_key"


class PycryptodomeAsymmetricSecurityAlgorithm(SecurityAlgorithm[bytes]):

    def __init__(self, algorithm: str = PycryptodomeAsymmetricAlgorithms.RSA, key_size: int = 2048,
                 min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.algorithm = algorithm.upper()
        self.key_size = key_size
        self.private_key = None
        self.public_key = None

    def _generate_and_save_key(self, key_file) -> KeyDetails:
        if self.private_key is not None and self.public_key is not None:
            raise Exception("Private and Public key are already initialized for PyCryptoDome RSA.")

        if self.algorithm == PycryptodomeAsymmetricAlgorithms.RSA:
            key = RSA.generate(self.key_size)
            self.private_key = key
            self.public_key = key.publickey()
        else:
            raise ValueError("Unsupported asymmetric algorithm in Pycryptodome.")

        with open(key_file, 'wb') as f:
            pickle.dump({
                PycryptodomeKeyConsts.PRIVATE_KEY: self.private_key.export_key(),
                PycryptodomeKeyConsts.PUBLIC_KEY: self.public_key.export_key()
            }, f)

        return KeyDetails(public_key={PycryptodomeKeyConsts.KEY_HOLDER: self.public_key},
                          private_key={PycryptodomeKeyConsts.KEY_HOLDER: self.private_key})

    def _load_key(self, key_file) -> KeyDetails:
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                data = pickle.load(f)
                if self.algorithm == PycryptodomeAsymmetricAlgorithms.RSA:
                    self.private_key = RSA.import_key(data[PycryptodomeKeyConsts.PRIVATE_KEY])
                    self.public_key = RSA.import_key(data[PycryptodomeKeyConsts.PUBLIC_KEY])
                else:
                    raise ValueError("Unsupported asymmetric algorithm in Pycryptodome.")
        else:
            raise Exception("Key file not found.")

        return KeyDetails(public_key={PycryptodomeKeyConsts.KEY_HOLDER: self.public_key},
                          private_key={PycryptodomeKeyConsts.KEY_HOLDER: self.private_key})

    def encrypt_message(self, msg: int) -> bytes:
        msg_bytes = str(msg).encode()
        if self.algorithm == PycryptodomeAsymmetricAlgorithms.RSA:
            cipher = PKCS1_OAEP.new(self.public_key)
            return cipher.encrypt(msg_bytes)
        else:
            raise NotImplementedError("Encryption is only supported for RSA.")

    def decrypt_message(self, msg: bytes) -> int:
        if self.algorithm == PycryptodomeAsymmetricAlgorithms.RSA:
            cipher = PKCS1_OAEP.new(self.private_key)
            plaintext = cipher.decrypt(msg)
            return int(plaintext.decode())
        else:
            raise NotImplementedError("Decryption is only supported for RSA.")
