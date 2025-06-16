import sys

from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils import convert_str_to_alg_type

NUMBER_OF_ARGUMENTS = 4
MESSAGES_FILE_INDEX = 1
ALGORITHM_INDEX = 2
ALGORITHM_KEY_INDEX = 3


def encrypt_messages(messages_file: str, encryption_algorithm: int, encryption_key_file: str) -> list[int]:
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
    encrypted_messages = []
    for message in messages_to_encrypt:
        encrypted_msg = encryption_class.encrypt_message(message)
        encrypted_messages.append(encrypted_msg)

    return encrypted_messages


if __name__ == "__main__":
    if len(sys.argv) < NUMBER_OF_ARGUMENTS:
        raise Exception("Usage: python encrypt_messages.py <messages_file> <encryption_algorithm> <key_file>")

    messages_file = sys.argv[MESSAGES_FILE_INDEX]
    encryption_algorithm = int(sys.argv[ALGORITHM_INDEX])
    encryption_key_file = sys.argv[ALGORITHM_KEY_INDEX]

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))

    encrypt_messages = encrypt_messages(messages_file, encryption_algorithm, encryption_key_file)
    print("Num of Encrypted Messages: {}".format(len(encrypt_messages)))

    print("First Encrypted Message: {}".format(encrypt_messages[0]))