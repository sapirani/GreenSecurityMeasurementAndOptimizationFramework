import argparse
import math
from typing import Optional, Any

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.encryption_type import EncryptionType, EncryptionMode
from tasks.confidential_computing_tasks.key_details import PRIME_MIN_VAL, PRIME_MAX_VAL
from tasks.confidential_computing_tasks.tasks_options.pipeline_parameters import PipelineParameters


def extract_arguments() -> PipelineParameters:
    parser = argparse.ArgumentParser(
        description="This program encrypts or decrypts messages using a security algorithm."
    )

    parser.add_argument("-m", "--messages_file",
                        type=str,
                        required=True,
                        help="path to messages file")

    parser.add_argument("-r", "--results_messages_file",
                        type=str,
                        required=True,
                        help="path to file for saving results")

    parser.add_argument("-a", "--algorithm",
                        type=int,
                        required=True,
                        help="type of encryption algorithm")

    parser.add_argument("--mode",
                        type=int,
                        default=None,
                        help="mode of block cipher algorithm")

    parser.add_argument("-k", "--key_file",
                        type=str,
                        required=True,
                        help="path to key file")

    parser.add_argument("--min_key_val",
                        type=int,
                        default=PRIME_MIN_VAL,
                        help="minimal key value")
    parser.add_argument("--max_key_val",
                        type=int,
                        default=PRIME_MAX_VAL,
                        help="maximal key value")

    args = parser.parse_args()

    messages_file = args.messages_file
    result_messages_file = args.results_messages_file
    encryption_algorithm = args.algorithm
    encryption_key_file = args.key_file
    min_key_val = args.min_key_val
    max_key_val = args.max_key_val
    cipher_block_mode = args.mode

    print("Messages File: {}".format(messages_file))
    print("Encryption Algorithm: {}".format(encryption_algorithm))
    return PipelineParameters(path_for_messages=messages_file,
                              path_for_result_messages=result_messages_file,
                              encryption_algorithm=encryption_algorithm,
                              cipher_block_mode=convert_mode_type_to_str(convert_int_to_mode_type(cipher_block_mode)),
                              key_file=encryption_key_file,
                              min_key_value=min_key_val,
                              max_key_value=max_key_val)


def convert_int_to_alg_type(encryption_algorithm: int) -> EncryptionType:
    try:
        encryption_type = EncryptionType(encryption_algorithm)
        return encryption_type
    except ValueError:
        raise Exception("Unsupported encryption algorithm.")

def convert_int_to_mode_type(encryption_mode: Optional[int]) -> Optional[EncryptionMode]:
    if encryption_mode is None:
        return None
    try:
        encryption_mode = EncryptionMode(encryption_mode)
        return encryption_mode
    except ValueError:
        raise Exception("Unsupported encryption mode.")

def convert_mode_type_to_str(mode: EncryptionMode) -> str:
    if mode is None:
        return None
    if mode == EncryptionMode.ECB:
        return "ECB"
    elif mode == EncryptionMode.CBC:
        return "CBC"
    elif mode == EncryptionMode.OFB:
        return "OFB"
    elif mode == EncryptionMode.CFB:
        return "CFB"
    elif mode == EncryptionMode.CTR:
        return "CTR"
    elif mode == EncryptionMode.GCM:
        return "GCM"
    elif mode == EncryptionMode.OCB:
        return "OCB"
    elif mode == EncryptionMode.OPENPGP:
        return "OpenPGP"
    elif mode == EncryptionMode.CCM:
        return "CCM"
    elif mode == EncryptionMode.EAX:
        return "EAX"
    elif mode == EncryptionMode.SIV:
        return "SIV"
    else:
        raise ValueError("Invalid block cipher mode")

def get_transformed_message(msg: int, action_type: ActionType, encryption_alg: SecurityAlgorithm) -> Any:
    if action_type == ActionType.Encryption:
        updated_msg = encryption_alg.encrypt_message(msg)

    elif action_type == ActionType.Decryption:
        updated_msg = encryption_alg.decrypt_message(msg)

    elif action_type == ActionType.FullPipeline:
        encrypted_message = encryption_alg.encrypt_message(msg)
        decrypted_message = encryption_alg.decrypt_message(encrypted_message)
        updated_msg = math.isclose(msg, decrypted_message, abs_tol=0.001)
    else:
        raise Exception("Unsupported action type.")
    return updated_msg


def is_new_execution(starting_messages_index: Optional[int]) -> bool:
    if starting_messages_index is None:
        return True
    if starting_messages_index <= 0:
        return True
    return False