import sys

from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils import convert_str_to_alg_type

NUMBER_OF_ARGUMENTS = 4
MESSAGES_FILE_INDEX = 1
ALGORITHM_INDEX = 2
ALGORITHM_KEY_INDEX = 3


def decrypt_messages(messages_file: str, decryption_algorithm: int, decryption_key_file: str) -> list[int]:
    encryption_algorithm_type = convert_str_to_alg_type(decryption_algorithm)

    try:
        with open(messages_file, "r") as messages_file:
            messages_to_decrypt = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages_to_decrypt) == 0:
        raise Exception("No messages to decrypt. Must be at least one message.")

    decryption_class = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type)
    key_details = decryption_class.extract_key(decryption_key_file)
    decrypted_messages = []
    for message in messages_to_decrypt:
        decrypted_msg = decryption_class.decrypt_message(message)
        decrypted_messages.append(decrypted_msg)

    return decrypted_messages


if __name__ == "__main__":
    if len(sys.argv) < NUMBER_OF_ARGUMENTS:
        raise Exception("Usage: python decrypt_messages.py <messages_file> <decryption_algorithm> <key_file>")

    messages_file = sys.argv[MESSAGES_FILE_INDEX]
    decryption_algorithm = int(sys.argv[ALGORITHM_INDEX])
    decryption_key_file = sys.argv[ALGORITHM_KEY_INDEX]

    print("Encrypted Messages File: {}".format(messages_file))
    print("Decryption Algorithm: {}".format(decryption_algorithm))

    decrypt_messages = decrypt_messages(messages_file, decryption_algorithm, decryption_key_file)
    print("Num of Decrypted Messages: {}".format(len(decrypt_messages)))
    print("First Decrypted Message: {}".format(decrypt_messages[0]))
