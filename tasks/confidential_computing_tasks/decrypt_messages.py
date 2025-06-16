import sys

from tasks.confidential_computing_tasks.algorithm_utils import get_messages_and_security_alg, extract_arguments


def decrypt_messages(messages_file: str, decryption_algorithm: int, decryption_key_file: str) -> list[int]:
    messages_to_decrypt, decryption_class = get_messages_and_security_alg(messages_file, decryption_algorithm, decryption_key_file)
    decrypted_messages = []
    for message in messages_to_decrypt:
        decrypted_msg = decryption_class.decrypt_message(message)
        decrypted_messages.append(decrypted_msg)

    return decrypted_messages


if __name__ == "__main__":
    messages_file, decryption_algorithm, decryption_key_file = extract_arguments()

    decrypt_messages = decrypt_messages(messages_file, decryption_algorithm, decryption_key_file)
    print("Num of Decrypted Messages: {}".format(len(decrypt_messages)))
    print("Decrypted Messages: {}".format(decrypt_messages))
