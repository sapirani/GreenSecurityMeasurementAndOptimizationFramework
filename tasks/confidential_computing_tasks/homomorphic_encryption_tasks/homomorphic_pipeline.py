import sys

from tasks.confidential_computing_tasks.algorithm_utils import get_messages_and_security_alg, extract_arguments


def use_homomorphic_encryption(messages_file: str, encryption_algorithm: int, encryption_key_file: str) -> list[int]:
    messages_to_encrypt, encryption_class = get_messages_and_security_alg(messages_file, encryption_algorithm)
    key_details = encryption_class.extract_key(encryption_key_file)
    valid_messages = []
    for message in messages_to_encrypt:
        encrypted_msg = encryption_class.encrypt_message(message)
        decrypted_msg = encryption_class.decrypt_message(encrypted_msg)
        valid_messages.append(message == decrypted_msg)

    return valid_messages


if __name__ == "__main__":
    messages_file, encryption_algorithm, encryption_key_file = extract_arguments(sys.argv, "homomorphic_pipeline")

    valid_messages = use_homomorphic_encryption(messages_file, encryption_algorithm, encryption_key_file)
    print("Num of Valid Messages: {}".format(len(valid_messages)))

    print("All Valid Messages: \n{}".format(valid_messages))
