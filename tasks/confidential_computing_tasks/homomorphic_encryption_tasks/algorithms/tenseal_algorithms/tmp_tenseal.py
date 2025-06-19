from typing import Union
import tenseal as ts
from tenseal import CKKSVector, BFVVector, Context

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails

EncryptedVector = Union[CKKSVector, BFVVector]

class TensealSecurityAlgorithm(HomomorphicSecurityAlgorithm[EncryptedVector]):
    def __init__(self, schema: str, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.schema = schema
        self._context = self._create_context_with_schema()

    def _create_context_with_schema(self) -> Context:
        if self.schema == "CKKS":
            context = ts.context(ts.SCHEME_TYPE.CKKS,
                                 poly_modulus_degree=8192,
                                 coeff_mod_bit_sizes=[60, 40, 40, 60]
                                 )
            context.global_scale = 2 ** 40

            # Enable encryption of data
            context.generate_galois_keys()
            return context
        elif self.schema == "BFV":
            return ts.context(ts.SCHEME_TYPE.BFV,
                              poly_modulus_degree=8192,  # Higher degree → more security & depth
                              plain_modulus=65537  # Prime modulus > max value in plaintext data
                              )
        else:
            print("ERROR")

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for tenseal library.")
        return KeyDetails({}, {})

    def encrypt_message(self, msg: int) -> EncryptedVector:
        """
        Encrypt the message
        Supporting only encrypting single message
        """
        data = [msg]
        return ts.ckks_vector(self._context, data)

    def decrypt_message(self, msg: EncryptedVector) -> int:
        """
        Decrypt the message
        Supporting only decrypting single message
        """
        return int(msg.decrypt()[0])

    def add_messages(self, c1: EncryptedVector, c2: EncryptedVector) -> EncryptedVector:
        try:
            return c1 + c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support adding messages.")

    def multiply_messages(self, c1: EncryptedVector, c2: EncryptedVector) -> EncryptedVector:
        try:
            return c1 * c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: EncryptedVector, scalar: int) -> EncryptedVector:
        try:
            return scalar * c
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying message with scalar.")
