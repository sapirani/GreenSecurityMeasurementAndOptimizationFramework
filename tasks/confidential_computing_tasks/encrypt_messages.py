import sys

from tasks.confidential_computing_tasks.algorithm_utils import get_messages_and_security_alg, extract_arguments


def encrypt_messages(messages_file: str, encryption_algorithm: int, encryption_key_file: str) -> list[int]:
    messages_to_encrypt, encryption_instance = get_messages_and_security_alg(messages_file, encryption_algorithm, encryption_key_file)
    encrypted_messages = []
    for message in messages_to_encrypt:
        encrypted_msg = encryption_instance.encrypt_message(message)
        encrypted_messages.append(encrypted_msg)

    return encrypted_messages


if __name__ == "__main__":
    messages_file, encryption_algorithm, encryption_key_file = extract_arguments()

    encrypt_messages = encrypt_messages(messages_file, encryption_algorithm, encryption_key_file)
    print("Num of Encrypted Messages: {}".format(len(encrypt_messages)))

    print("Encrypted Messages: {}".format(encrypt_messages))
