import signal
import sys
import types
from functools import partial

from selenium.webdriver.support.expected_conditions import element_selection_state_to_be

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, convert_int_to_alg_type, \
    get_updated_message, is_new_execution
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file, \
    write_messages_to_file, write_last_message_index, get_last_message_index


class StorageForExit:
    def __init__(self, alg: SecurityAlgorithm, results_path: str, updated_messages: list, action_type: ActionType,
                 initial_message_index: int):
        self.alg = alg
        self.results_path = results_path
        self.updated_messages = updated_messages
        self.action_type = action_type
        self.should_override = is_new_execution(initial_message_index)
        self.initial_message_index = initial_message_index

    @property
    def last_message_index(self):
        return len(self.updated_messages) + self.initial_message_index

    def save_updated_messages(self):
        if self.last_message_index is not None and self.last_message_index > 0:
            write_last_message_index(self.last_message_index)
        if self.updated_messages is not None and len(self.updated_messages) > 0 and self.results_path != "":
            print("SHOULD OVERRIDE SAVING PATH (in class):", self.should_override)
            if self.action_type == ActionType.Encryption:
                self.alg.save_encrypted_messages(self.updated_messages, self.results_path, should_override_file=self.should_override)
            # If decryption or full pipeline, optionally save decrypted ints as text
            elif self.action_type in (ActionType.Decryption, ActionType.FullPipeline):
                write_messages_to_file(self.results_path, self.updated_messages, should_override_file=self.should_override)


def handle_sigint(sig, frame: types.FrameType, storage_for_exit: StorageForExit):
    print("Sub process Received SIGINT! Cleaning up...")
    storage_for_exit.save_updated_messages()
    sys.exit(0)

def get_message_for_pipeline(messages_file_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int) -> list:
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
    print("SHOULD OVERRIDE SAVING PATH (in main):", should_override)
    if action == ActionType.Encryption:
        alg.save_encrypted_messages(messages, results_path, should_override)
    # If decryption or full pipeline, optionally save decrypted ints as text
    elif action in (ActionType.Decryption, ActionType.FullPipeline):
        write_messages_to_file(results_path, messages, should_override)
    else:
        raise Exception("Unknown action type.")


def execute_regular_pipeline(action_type: ActionType) -> list[int]:
    params = extract_arguments()
    last_message_index = get_last_message_index()

    encryption_algorithm_type = convert_int_to_alg_type(params.encryption_algorithm)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               params.cipher_block_mode,
                                                                               params.min_key_value,
                                                                               params.max_key_value)
    encryption_instance.extract_key(params.key_file, should_generate=is_new_execution(last_message_index))
    updated_messages = []

    storage_for_exit = StorageForExit(alg=encryption_instance, results_path=params.path_for_result_messages,
                                      updated_messages=updated_messages, action_type=action_type,
                                      initial_message_index=last_message_index)
    signal.signal(signal.SIGBREAK, partial(handle_sigint, storage_for_exit=storage_for_exit))

    messages = get_message_for_pipeline(params.path_for_messages, encryption_instance, action_type, last_message_index)
    print(f"After snip messages: {len(messages)}")
    for message in messages:
        updated_msg = get_updated_message(message, action_type, encryption_instance)
        updated_messages.append(updated_msg)
    save_messages_for_pipeline(messages, params.path_for_result_messages, encryption_instance, action_type, last_message_index)

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
