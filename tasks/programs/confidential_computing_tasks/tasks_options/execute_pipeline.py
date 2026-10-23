import signal
import threading
from functools import partial
from typing import Optional

from tasks.programs.confidential_computing_tasks.abstract_security_algorithm import SecurityAlgorithm
from tasks.programs.confidential_computing_tasks.action_type import ActionType
from tasks.programs.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.programs.confidential_computing_tasks.key_details import KeyDetails
from tasks.programs.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, \
    convert_int_to_alg_type, \
    get_transformed_message, is_new_execution
from tasks.programs.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_operation_storage import \
    OperationCheckpointStorage
from tasks.programs.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_storage import CheckpointStorage
from tasks.programs.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file, \
    write_messages_to_file, get_last_message_index, read_checkpoint_file

checkpoint_storage: Optional[OperationCheckpointStorage] = None


def checkpoint_callback(curren_encrypted_messages, total):
    if checkpoint_storage:
        checkpoint_storage.update(curren_encrypted_messages, total)
        checkpoint_storage.save_checkpoint()


def handle_signal(signum, frame, storage: CheckpointStorage, done_event: threading.Event):
    if storage:
        print("\n[Signal received] Saving checkpoint...")
        done_event.set()


def get_message(messages_file_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int) -> list:
    """
    Read messages from a file.
    For encryption, multiplication and addition (of encrypted data) operations - read plain text messages.
    For decryption operations - read encrypted messages.
    Input:
        - messages_file_path: Path to the messages file.
        - alg: The cryptographic algorithm to use.
        - action: what type of operation to perform on the messages.
        - starting_index: Index of the first message to read.
    Output:
        - list of messages read from the file.
    """
    if action == ActionType.Encryption or action == ActionType.FullPipeline or action == ActionType.Addition or action == ActionType.Multiplication:
        messages = extract_messages_from_file(messages_file_path)
    elif action == ActionType.Decryption:
        messages = alg.load_encrypted_messages(messages_file_path, starting_index)
    else:
        messages = []

    return messages


def save_messages_for_pipeline(messages: list, results_path: str, alg: SecurityAlgorithm, action: ActionType,
                               should_override: bool):
    """
    Save the messages after performing the given operation on them using the given algorithm.
    Input:
        - messages: List of messages to save.
        - results_path: Path to the file where the messages should be saved.
        - action: The type of operation that was performed on the messages.
        - starting_index: Index of the first message that was read.
    """
    if action == ActionType.Encryption:
        alg.save_encrypted_messages(messages, results_path, should_override)
    # If decryption or full pipeline, optionally save decrypted ints as text
    elif action in (ActionType.Decryption, ActionType.FullPipeline):
        write_messages_to_file(results_path, messages, should_override)
    else:
        raise Exception("Unknown action type.")


def extract_key_for_algorithm(key_file_path: str, alg: SecurityAlgorithm, action: ActionType,
                              starting_index: int) -> KeyDetails:
    if action == ActionType.Decryption:
        return alg.extract_key(key_file_path, should_generate=False)
    return alg.extract_key(key_file_path, should_generate=is_new_execution(starting_index))


def execute_regular_pipeline(action_type: ActionType) -> list[int]:
    """
    Execute regular encryption or decryption pipeline.
    """
    params = extract_arguments()
    last_message_index = get_last_message_index()

    # get the relevant algorithm
    encryption_algorithm_type = convert_int_to_alg_type(params.encryption_algorithm)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               params.cipher_block_mode,
                                                                               params.min_key_value,
                                                                               params.max_key_value)

    transformed_messages = []

    done_event = threading.Event()

    # create storage for saving state in cases of crashing in the middle of an experiment
    storage = CheckpointStorage(alg=encryption_instance, results_path=params.path_for_result_messages,
                                transformed_messages=transformed_messages, action_type=action_type,
                                initial_message_index=last_message_index)

    # define signals that should be caught using the done_event
    signal.signal(signal.SIGBREAK, partial(handle_signal, storage=storage, done_event=done_event))
    signal.signal(signal.SIGTERM, partial(handle_signal, storage=storage, done_event=done_event))

    extract_key_for_algorithm(params.key_file, encryption_instance, action_type, last_message_index)

    # get messages and perform the operation on each one
    messages = get_message(params.path_for_messages, encryption_instance, action_type, last_message_index)
    for message in messages:
        transformed_msg = get_transformed_message(message, action_type, encryption_instance)
        transformed_messages.append(transformed_msg)

        if done_event.is_set():
            storage.transformed_messages = transformed_messages
            storage.save_checkpoint()
            break


    should_override = is_new_execution(last_message_index)
    save_messages_for_pipeline(transformed_messages, params.path_for_result_messages, encryption_instance, action_type,
                               should_override)

    return transformed_messages


def execute_operation(messages: list[int], action: ActionType, algorithm: SecurityAlgorithm, total_checkpoint,
                      done_event: threading.Event) -> int:
    """
    Perform a cryptographic homomorphic operation on messages. Can be addition or multiplication.
    After performing the operation, decrypt the result.
    Input:
        - messages: Messages to perform this operation on.
        - action: The type of operation to perform.
        - algorithm: The cryptographic algorithm to use. Should be Homomorphic algorithm.
        - total_checkpoint: The
    """
    if total_checkpoint:
        deserialized_total = algorithm.deserialize_message(total_checkpoint)
    else:
        deserialized_total = None

    if action == ActionType.Addition:
        encrypted_res = algorithm.calc_encrypted_sum(messages, done_event=done_event, start_total=deserialized_total,
                                                     checkpoint_callback=checkpoint_callback)
    elif action == ActionType.Multiplication:
        encrypted_res = algorithm.calc_encrypted_multiplication(messages, done_event=done_event,
                                                                start_total=deserialized_total,
                                                                checkpoint_callback=checkpoint_callback)
    else:
        raise Exception("Unknown action type.")
    return algorithm.decrypt_message(encrypted_res)


def execute_homomorphic_pipeline(action_type: ActionType) -> int:
    """
    A method for executing homomorphic operation (add or multiply).
    For homomorphic algorithms -> first encrypt, then run the operation and then decrypt.
    For traditional algorithms -> first run the operation, then encrypt and decrypt.
    """
    global checkpoint_storage

    params = extract_arguments()
    last_message_index, total_checkpoint = read_checkpoint_file()

    encryption_algorithm_type = convert_int_to_alg_type(params.encryption_algorithm)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(
        encryption_algorithm_type,
        params.cipher_block_mode,
        params.min_key_value,
        params.max_key_value
    )

    transformed_messages = []

    done_event = threading.Event()
    checkpoint_storage = OperationCheckpointStorage(
        alg=encryption_instance,
        results_path=params.path_for_result_messages,
        transformed_messages=transformed_messages,
        action_type=action_type,
        initial_message_index=last_message_index
    )

    signal.signal(signal.SIGBREAK, partial(handle_signal, storage=checkpoint_storage, done_event=done_event))
    signal.signal(signal.SIGTERM, partial(handle_signal, storage=checkpoint_storage, done_event=done_event))

    extract_key_for_algorithm(params.key_file, encryption_instance, action_type, last_message_index)

    messages = get_message(params.path_for_messages, encryption_instance, action_type, last_message_index)

    operation_encrypted_result = execute_operation(messages, action_type, encryption_instance, total_checkpoint,
                                                   done_event)
    print(f"The original messages: {messages}")
    print(f"The decrypted result: {operation_encrypted_result}")
    return operation_encrypted_result
