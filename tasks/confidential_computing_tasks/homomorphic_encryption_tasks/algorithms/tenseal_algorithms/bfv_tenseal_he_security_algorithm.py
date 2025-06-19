import tenseal as ts
from tenseal import Context, BFVVector

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.algorithms.tenseal_algorithms.tenseal_he_security_algorithm import \
    TensealSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL


class BFVTensealSecurityAlgorithm(TensealSecurityAlgorithm[BFVVector]):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        self.__plain_modulus = 65537
        super().__init__(min_key_val, max_key_val)

    def _create_context_with_schema(self) -> Context:
        return ts.context(ts.SCHEME_TYPE.BFV,
                          poly_modulus_degree=8192,  # Higher degree → more security & depth
                          plain_modulus=self.__plain_modulus # Prime modulus > max value in plaintext data
                          )

    def encrypt_message(self, msg: int) -> BFVVector:
        """
        Encrypt the message
        Supporting only encrypting single message
        """
        data = [msg]
        return ts.bfv_vector(self._context, data)

    def decrypt_message(self, msg: BFVVector) -> int:
        """
        Decrypt the message
        Supporting only decrypting single message
        """
        decrypted_messages = msg.decrypt()

        # fix for v such that v > plain_modulus // 2 is interpreted as v - plain_modulus.
        updated_messages = [(x + self.__plain_modulus) % self.__plain_modulus for x in decrypted_messages]
        return int(updated_messages[0])

    def add_messages(self, c1: BFVVector, c2: BFVVector) -> BFVVector:
        try:
            return c1 + c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support adding messages.")

    def multiply_messages(self, c1: BFVVector, c2: BFVVector) -> BFVVector:
        try:
            return c1 * c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: BFVVector, scalar: int) -> BFVVector:
        try:
            return scalar * c
        except Exception as e:
            raise NotImplementedError(f"tenseal with CKKS schema does not support multiplying message with scalar.")

if __name__ == '__main__':
    m1 = 56
    m2 = 84
    ten = BFVTensealSecurityAlgorithm()

    e1_vec = ten.encrypt_message(m1)
    e2_vec = ten.encrypt_message(m2)

    enc_result = ten.add_messages(e1_vec, e2_vec)
    enc_result = ten.scalar_and_message_multiplication(enc_result, 3)
    enc_result = ten.multiply_messages(enc_result, e2_vec)

    # Decrypt and print result
    decrypted = ten.decrypt_message(enc_result)
    print("Decrypted result:", decrypted)
