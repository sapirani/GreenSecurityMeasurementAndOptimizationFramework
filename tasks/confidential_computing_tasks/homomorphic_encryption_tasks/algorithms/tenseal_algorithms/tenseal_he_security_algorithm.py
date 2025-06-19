from abc import ABC, abstractmethod
from typing import Union

import tenseal as ts
from tenseal import Context, BFVVector, CKKSVector

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import T
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL


class TensealSchemas:
    CKKS = "CKKS"
    BFV = "BFV"


class TensealSecurityAlgorithm(HomomorphicSecurityAlgorithm[T], ABC):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self._context = self._create_context_with_schema()

    @abstractmethod
    def _create_context_with_schema(self) -> Context:
        pass

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for tenseal library.")
        return KeyDetails({}, {})



# if __name__ == '__main__':
#     m1 = 1
#     m2 = 2
#     ten = TensealSecurityAlgorithm(TensealSchemas.CKKS)
#
#     e1_vec = ten.encrypt_message(m1)
#     e2_vec = ten.encrypt_message(m2)
#
#     enc_result = ten.add_messages(e1_vec, e2_vec)
#     enc_result = ten.scalar_and_message_multiplication(enc_result, 3)
#     enc_result = ten.multiply_messages(enc_result, e2_vec)
#
#     # Decrypt and print result
#     decrypted = ten.decrypt_message(enc_result)
#     print("Decrypted result:", decrypted)