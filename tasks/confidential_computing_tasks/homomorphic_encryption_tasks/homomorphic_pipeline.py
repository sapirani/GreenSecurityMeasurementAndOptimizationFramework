import sys

from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils import convert_str_to_alg_type

NUMBER_OF_ARGUMENTS = 4
MESSAGES_FILE_INDEX = 1
ALGORITHM_INDEX = 2
ALGORITHM_KEY_INDEX = 3


def use_homomorphic_encryption(messages_file: str, encryption_algorithm: int, encryption_key_file: str) -> list[int]:
    encryption_algorithm_type = convert_str_to_alg_type(encryption_algorithm)

    try:
        with open(messages_file, "r") as messages_file:
            messages_to_encrypt = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages_to_encrypt) == 0:
        raise Exception("No messages to encrypt. Must be at least one message.")

    encryption_class = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type)
    key_details = encryption_class.extract_key(encryption_key_file)
    valid_messages = []
    for message in messages_to_encrypt:
        encrypted_msg = encryption_class.encrypt_message(message)
        decrypted_msg = encryption_class.decrypt_message(encrypted_msg)
        valid_messages.append(message == decrypted_msg)

    return valid_messages


if __name__ == "__main__":
    if len(sys.argv) < NUMBER_OF_ARGUMENTS:
        raise Exception("Usage: python encrypt_messages.py <messages_file> <encryption_algorithm> <key_file>")

    messages_file = sys.argv[MESSAGES_FILE_INDEX]
    encryption_algorithm = int(sys.argv[ALGORITHM_INDEX])
    encryption_key_file = sys.argv[ALGORITHM_KEY_INDEX]

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))

    valid_messages = use_homomorphic_encryption(messages_file, encryption_algorithm, encryption_key_file)
    print("Num of Valid Messages: {}".format(len(valid_messages)))

    print("All Valid Messages: \n{}".format(valid_messages))