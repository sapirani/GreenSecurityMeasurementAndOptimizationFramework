import pickle

from concrete import fhe
from concrete.compiler import KeySet
from concrete.fhe import Value

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL, KeyDetails


@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def add(x, y):
    return x + y


# Function 2: Multiply two encrypted integers
@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def multiply(x, y):
    return x * y


# Function 3: Multiply encrypted integer by scalar (constant)
@fhe.compiler({"x": "encrypted", "scalar": "clear"})
def scalar_multiply(x, scalar):
    return x * scalar


class ConcreteFHESecurityAlgorithm(HomomorphicSecurityAlgorithm[fhe.Value]):
    __KEYS_FILE = "encryption_model.bin"

    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        super().__init__(min_key_val, max_key_val)
        inputset = [(10, 185), (9, 200), (15, 198), (15, 230)] # todo: change?

        self.__config = fhe.Configuration(
            # you can inset here so many configurations
            p_error=0.001,  # Optional: sets per-lookup error probability
            dataflow_parallelize=True  # Optional: parallelize computation
        )

        # Define the same keys for all circuits
        self.__add_circuit = add.compile(inputset, configuration=self.__config)  # Dummy inputs for compilation
        self.__keyset = self.__add_circuit.keys

        self.__mul_circuit = multiply.compile(inputset, configuration=self.__config)  # Dummy inputs for compilation
        self.__mul_circuit.keys = self.__keyset

        self.__mul_with_scalar_circuit = scalar_multiply.compile(inputset, configuration=self.__config)  # Dummy inputs for compilation
        self.__mul_with_scalar_circuit.keys = self.__keyset

    def extract_key(self, key_file: str) -> KeyDetails:  # todo: maybe extract from the key file the alg name?
        """ Initialize the public and private key """
        print("Key extraction method is not implemented for Concrete-python fhe library.")
        return KeyDetails({}, {})

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[fhe.Value]) -> list[fhe.Value]:
        try:
            with open(self.__KEYS_FILE, "wb") as f:
                f.write(self.__keyset.serialize())
            # self.__add_circuit.keyset.save(self.__KEYS_FILE)
            return [self.__add_circuit.serialize_encrypted_input(ciphertext) for ciphertext in encrypted_messages]
        except Exception as e:
            raise RuntimeError("Error occurred when saving lightPhe model.")

    def _get_deserializable_encrypted_messages(self, encrypted_messages: list[fhe.Value]) -> list[fhe.Value]:
        try:
            with open(self.__KEYS_FILE, "rb") as f:
                self.__keyset = KeySet.deserialize(pickle.load(f))
            self.__add_circuit.keys = self.__keyset
            self.__mul_circuit.keys = self.__keyset
            self.__mul_with_scalar_circuit.keys = self.__keyset
            return [self.__add_circuit.deserialize_encrypted_input(ciphertext_bytes) for ciphertext_bytes in
                    encrypted_messages]
        except FileNotFoundError:
            print("Something went wrong with loading the encrypted messages")

        return encrypted_messages

    def encrypt_message(self, msg: int) -> fhe.Value:
        """ Encrypt the message """
        return self.__add_circuit.encrypt_input(x=msg)["x"]

    def decrypt_message(self, msg: fhe.Value) -> int:
        """ Decrypt the message """
        return self.__add_circuit.decrypt_output(msg)

    def add_messages(self, c1: fhe.Value, c2: fhe.Value) -> fhe.Value:
        try:
            return self.__add_circuit.run({"x": c1, "y": c2})
        except Exception as e:
            raise NotImplementedError(f"Concrete-python fhe does not support adding messages.")

    def multiply_messages(self, c1: fhe.Value, c2: fhe.Value) -> fhe.Value:
        try:
            return self.__mul_circuit.run({"x": c1, "y": c2})
        except Exception as e:
            raise NotImplementedError(
                f"Concrete-python fhe does not support multiplying messages.")

    def scalar_and_message_multiplication(self, c: fhe.Value, scalar: int) -> fhe.Value:
        try:
            return self.__mul_with_scalar_circuit.run({"x": c, "scalar": scalar})
        except Exception as e:
            raise NotImplementedError(
                f"Concrete-python fhe does not support multiplying message with scalar.")

if __name__ == "__main__":
    co = ConcreteFHESecurityAlgorithm()
    a = 56
    b = 83

    c1 = co.encrypt_message(a)
    c2 = co.encrypt_message(b)

    co.save_encrypted_messages([c1, c2], "temp.bin")
