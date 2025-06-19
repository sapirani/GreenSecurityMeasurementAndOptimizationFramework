import tenseal as ts
from tenseal import Context

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import T
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


ERROR_NOT_SUPPORTING_SCHEMA = "TenSEAL library supports only CKKS and BFV schemas."

class TensealSchemas:
    CKKS = "CKKS"
    BFV = "BFV"


class TensealSecurityAlgorithm(HomomorphicSecurityAlgorithm[T]):
    def __init__(self, schema: str, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        self.schema = schema
        self.__plain_modulus = 65537
        self.__plain_modulus_deg = 8192
        self.__coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.__global_scale = 2 ** 40
        self.__context = self._create_context_with_schema()

    def _create_context_with_schema(self) -> Context:
        if self.schema == TensealSchemas.CKKS:
            context = ts.context(ts.SCHEME_TYPE.CKKS,
                                 poly_modulus_degree=self.__plain_modulus_deg,
                                 coeff_mod_bit_sizes=self.__coeff_mod_bit_sizes
                                 )
            context.global_scale = self.__global_scale

            # Enable encryption of data
            context.generate_galois_keys()
            return context
        elif self.schema == TensealSchemas.BFV:
            return ts.context(ts.SCHEME_TYPE.BFV,
                              poly_modulus_degree=self.__plain_modulus_deg,  # Higher degree → more security & depth
                              plain_modulus=self.__plain_modulus  # Prime modulus > max value in plaintext data
                              )
        else:
            raise NotImplementedError(ERROR_NOT_SUPPORTING_SCHEMA)

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for tenseal library.")
        return KeyDetails({}, {})

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[T]) -> list[T]:
        return [msg.serialize() for msg in encrypted_messages]

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[T]) -> list[T]:
        return [T.deserialize(self.__context, msg) for msg in encrypted_messages]

    def encrypt_message(self, msg: int) -> T:
        """
        Encrypt the message
        Supporting only encrypting single message
        """
        data = [msg]
        if self.schema == TensealSchemas.CKKS:
            return ts.ckks_vector(self.__context, data)
        elif self.schema == TensealSchemas.BFV:
            return ts.bfv_vector(self.__context, data)
        else:
            raise NotImplementedError(ERROR_NOT_SUPPORTING_SCHEMA)

    def decrypt_message(self, msg: T) -> int:
        """
        Decrypt the message
        Supporting only decrypting single message
        msg contains a single encrypted message
        """

        decrypted_message = msg.decrypt()[0]
        if self.schema == TensealSchemas.CKKS:
            return decrypted_message
        elif self.schema == TensealSchemas.BFV:
            return (decrypted_message + self.__plain_modulus) % self.__plain_modulus
        else:
            raise NotImplementedError(ERROR_NOT_SUPPORTING_SCHEMA)

    def add_messages(self, c1: T, c2: T) -> T:
        try:
            return c1 + c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with {self.schema} schema does not support adding messages.")

    def multiply_messages(self, c1: T, c2: T) -> T:
        try:
            return c1 * c2
        except Exception as e:
            raise NotImplementedError(f"tenseal with {self.schema} schema does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: T, scalar: int) -> T:
        try:
            return scalar * c
        except Exception as e:
            raise NotImplementedError(
                f"tenseal with {self.schema} schema does not support multiplying message with scalar.")
