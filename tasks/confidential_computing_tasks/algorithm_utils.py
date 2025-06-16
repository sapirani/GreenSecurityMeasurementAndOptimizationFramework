from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.encryption_type import EncryptionType

NUMBER_OF_ARGUMENTS = 4
MESSAGES_FILE_INDEX = 1
ALGORITHM_INDEX = 2
ALGORITHM_KEY_INDEX = 3

def extract_arguments(arguments: list[str]) -> tuple[str, int, str]:
    if len(arguments) < NUMBER_OF_ARGUMENTS:
        raise Exception("Usage: python encrypt_messages.py <messages_file> <encryption_algorithm> <key_file>")

    messages_file = arguments[MESSAGES_FILE_INDEX]
    encryption_algorithm = int(arguments[ALGORITHM_INDEX])
    encryption_key_file = arguments[ALGORITHM_KEY_INDEX]

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))
    return messages_file, encryption_algorithm, encryption_key_file


def convert_str_to_alg_type(encryption_algorithm: int) -> EncryptionType:
    try:
        encryption_type = EncryptionType(encryption_algorithm)
        return encryption_type
    except ValueError:
        raise Exception("Unsupported encryption algorithm.")


def get_messages_and_security_alg(messages_file: str, encryption_algorithm: int) -> tuple[list[int], SecurityAlgorithm]:
    encryption_algorithm_type = convert_str_to_alg_type(encryption_algorithm)

    try:
        with open(messages_file, "r") as messages_file:
            messages = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages) == 0:
        raise Exception("No messages to encrypt. Must be at least one message.")

    encryption_class = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type)
    return messages, encryption_class
