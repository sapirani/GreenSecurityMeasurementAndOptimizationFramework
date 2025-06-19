import tenseal as ts
from tenseal import Context, CKKSVector

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.tenseal_algorithms.tenseal_he_security_algorithm import \
    TensealSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL


class CKKSTensealSecurityAlgorithm(TensealSecurityAlgorithm[CKKSVector]):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        self.__coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.__poly_modulus_degree = 8192
        self.__global_scale = 2 ** 40
        super().__init__(min_key_val, max_key_val)

    def _create_context_with_schema(self) -> Context:
        context = ts.context(ts.SCHEME_TYPE.CKKS,
                             poly_modulus_degree=self.__poly_modulus_degree,
                             coeff_mod_bit_sizes=self.__coeff_mod_bit_sizes
                             )
        context.global_scale = self.__global_scale

        # Enable encryption of data
        context.generate_galois_keys()
        return context

    def encrypt_message(self, msg: int) -> CKKSVector:
        """
        Encrypt the message
        Supporting only encrypting single message
        """
        data = [msg]
        return ts.ckks_vector(self._context, data)

    def decrypt_message(self, msg: CKKSVector) -> int:
        """
        Decrypt the message
        Supporting only decrypting single message
        """
        return int(msg.decrypt()[0])

    def add_messages(self, c1: CKKSVector, c2: CKKSVector) -> CKKSVector:
        try:
            return c1 + c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support adding messages.")

    def multiply_messages(self, c1: CKKSVector, c2: CKKSVector) -> CKKSVector:
        try:
            return c1 * c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: CKKSVector, scalar: int) -> CKKSVector:
        try:
            return scalar * c
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying message with scalar.")
