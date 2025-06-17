import argparse

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL


def extract_arguments() -> tuple[str, int, str, int, int]:
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

    parser.add_argument("--min",
                        type=int,
                        default=PRIME_MIN_VAL,
                        help="minimal key value")
    parser.add_argument("--max",
                        type=int,
                        default=PRIME_MAX_VAL,
                        help="maximal key value")

    args = parser.parse_args()

    messages_file = args.messages_file
    encryption_algorithm = args.algorithm
    encryption_key_file = args.key_file

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))
    return messages_file, encryption_algorithm, encryption_key_file, args.min, args.max


def convert_str_to_alg_type(encryption_algorithm: int) -> EncryptionType:
    try:
        encryption_type = EncryptionType(encryption_algorithm)
        return encryption_type
    except ValueError:
        raise Exception("Unsupported encryption algorithm.")


def extract_messages_from_file(messages_file: str) -> list[int]:
    messages = []
    try:
        with open(messages_file, "r") as messages_file:
            messages = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages) == 0:
        raise Exception("No messages found. Must be at least one message.")

    return messages


def get_updated_message(msg: int, action_type: ActionType, encryption_alg: SecurityAlgorithm) -> int:
    if action_type == ActionType.Encryption:
        updated_msg = encryption_alg.encrypt_message(msg)

    elif action_type == ActionType.Decryption:
        updated_msg = encryption_alg.decrypt_message(msg)

    elif action_type == ActionType.FullPipeline:
        encrypted_message = encryption_alg.encrypt_message(msg)
        decrypted_message = encryption_alg.decrypt_message(encrypted_message)
        updated_msg = int(decrypted_message == msg)
    else:
        raise Exception("Unsupported action type.")
    return updated_msg
