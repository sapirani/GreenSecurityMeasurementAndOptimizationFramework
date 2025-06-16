import argparse

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.encryption_type import EncryptionType


def extract_arguments() -> tuple[str, int, str]:
    parser = argparse.ArgumentParser(
        description="This program encrypts or decrypts messages using a security algorithm."
    )

    parser.add_argument("-m", "--messages_file",
                        type=str,
                        required=True,
                        help="path to messages file")

    parser.add_argument("-a", "--algorithm",
                        type=int,
                        required=True,
                        help="type of encryption algorithm")

    parser.add_argument("-k", "--key_file",
                        type=str,
                        required=True,
                        help="path to key file")

    args = parser.parse_args()

    messages_file = args.messages_file
    encryption_algorithm = args.algorithm
    encryption_key_file = args.key_file

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))
    return messages_file, encryption_algorithm, encryption_key_file


def convert_str_to_alg_type(encryption_algorithm: int) -> EncryptionType:
    try:
        encryption_type = EncryptionType(encryption_algorithm)
        return encryption_type
    except ValueError:
        raise Exception("Unsupported encryption algorithm.")


def get_messages_and_security_alg(messages_file: str, encryption_algorithm: int, encryption_key_file: str) -> tuple[
    list[int], SecurityAlgorithm]:
    encryption_algorithm_type = convert_str_to_alg_type(encryption_algorithm)

    try:
        with open(messages_file, "r") as messages_file:
            messages = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages) == 0:
        raise Exception("No messages found. Must be at least one message.")

    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type)
    encryption_instance.extract_key(encryption_key_file)
    return messages, encryption_instance
