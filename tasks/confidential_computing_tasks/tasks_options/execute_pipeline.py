import signal
import sys
import types
from functools import partial

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.key_details import KeyDetails
from tasks.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, convert_int_to_alg_type, \
    get_transformed_message, is_new_execution
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file, \
    write_messages_to_file, get_last_message_index
from tasks.confidential_computing_tasks.utils.storage_for_exit import Storage


def handle_sigint(sig, frame: types.FrameType, storage_for_exit: Storage):
    print("Sub process Received SIGINT! Cleaning up...")
    storage_for_exit.save_updated_messages()
    sys.exit(0)

def get_message(messages_file_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int) -> list:
    if action == ActionType.Encryption or action == ActionType.FullPipeline:
        messages = extract_messages_from_file(messages_file_path)
    elif action == ActionType.Decryption:
        messages = alg.load_encrypted_messages(messages_file_path)
    else:
        messages = []

    if starting_index >= len(messages):
        return messages
    return messages[starting_index:]

def save_messages_for_pipeline(messages: list, results_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int):
    should_override = is_new_execution(starting_index)
    if action == ActionType.Encryption:
        alg.save_encrypted_messages(messages, results_path, should_override)
    # If decryption or full pipeline, optionally save decrypted ints as text
    elif action in (ActionType.Decryption, ActionType.FullPipeline):
        write_messages_to_file(results_path, messages, should_override)
    else:
        raise Exception("Unknown action type.")

def extract_key_for_algorithm(key_file_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int) -> KeyDetails:
    if action == ActionType.Decryption:
        return alg.extract_key(key_file_path, should_generate=False)
    return alg.extract_key(key_file_path, should_generate=is_new_execution(starting_index))


def execute_regular_pipeline(action_type: ActionType) -> list[int]:
    params = extract_arguments()
    last_message_index = get_last_message_index()

    encryption_algorithm_type = convert_int_to_alg_type(params.encryption_algorithm)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               params.cipher_block_mode,
                                                                               params.min_key_value,
                                                                               params.max_key_value)

    extract_key_for_algorithm(params.key_file, encryption_instance, action_type, last_message_index)
    updated_messages = []

    storage_for_exit = Storage(alg=encryption_instance, results_path=params.path_for_result_messages,
                               updated_messages=updated_messages, action_type=action_type,
                               initial_message_index=last_message_index)
    signal.signal(signal.SIGBREAK, partial(handle_sigint, storage_for_exit=storage_for_exit))
    signal.signal(signal.SIGTERM, partial(handle_sigint, storage_for_exit=storage_for_exit))

    messages = get_message(params.path_for_messages, encryption_instance, action_type, last_message_index)
    for message in messages:
        updated_msg = get_transformed_message(message, action_type, encryption_instance)
        updated_messages.append(updated_msg)
    save_messages_for_pipeline(updated_messages, params.path_for_result_messages, encryption_instance, action_type, last_message_index)

    return updated_messages


def execute_operation(messages: list[int], action: ActionType, algorithm: SecurityAlgorithm) -> int:
    if action == ActionType.Addition:
        encrypted_res = algorithm.calc_encrypted_sum(messages)
    elif action == ActionType.Multiplication:
        encrypted_res = algorithm.calc_encrypted_multiplication(messages)
    else:
        raise Exception("Unknown encryption action type.")
    return algorithm.decrypt_message(encrypted_res)


def execute_homomorphic_pipeline(action_type: ActionType) -> int:
    """
    A method for executing homomorphic operation (add or multiply).
    For homomorphic algorithms -> first encrypt, then run the operation and then decrypt.
    For traditional algorithms -> first run the operation, then encrypt and decrypt.
    """
    # TODO: add saving of messages
    params = extract_arguments()
    messages_file = params.path_for_messages
    encryption_algorithm = params.encryption_algorithm
    encryption_key_file = params.key_file
    min_key_val = params.min_key_value
    max_key_val = params.max_key_value
    cipher_block_mode = params.cipher_block_mode

    encryption_algorithm_type = convert_int_to_alg_type(encryption_algorithm)

    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               cipher_block_mode,
                                                                               min_key_val,
                                                                               max_key_val)

    encryption_instance.extract_key(encryption_key_file)
    messages = extract_messages_from_file(messages_file)

    operation_encrypted_result = execute_operation(messages, action_type, encryption_instance)
    print(f"The original messages: {messages}")
    print(f"The decrypted result: {operation_encrypted_result}")
    return operation_encrypted_result
