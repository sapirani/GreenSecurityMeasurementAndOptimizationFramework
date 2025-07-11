import signal
from functools import partial
from typing import Optional

from tasks.confidential_computing_tasks.abstract_security_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.key_details import KeyDetails
from tasks.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, convert_int_to_alg_type, \
    get_transformed_message, is_new_execution
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_operation_storage import \
    OperationCheckpointStorage
from tasks.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_storage import CheckpointStorage
from tasks.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file, \
    write_messages_to_file, get_last_message_index, save_checkpoint_file, read_checkpoint_file

checkpoint_storage: Optional[OperationCheckpointStorage] = None


def checkpoint_callback(curren_encrypted_msg, total):
    if checkpoint_storage:
        checkpoint_storage.update(curren_encrypted_msg, total)


def handle_signal(signum, frame, storage: CheckpointStorage):
    if storage:
        print("\n[Signal received] Saving checkpoint...")
        storage.save_checkpoint()
    exit(0)


def get_message(messages_file_path: str, alg: SecurityAlgorithm, action: ActionType, starting_index: int) -> list:
    if action == ActionType.Encryption or action == ActionType.FullPipeline or action == ActionType.Addition or action == ActionType.Multiplication:
        messages = extract_messages_from_file(messages_file_path)
    elif action == ActionType.Decryption:
        messages = alg.load_encrypted_messages(messages_file_path)
    else:
        messages = []

    if starting_index >= len(messages):
        return messages
    return messages[starting_index:]


def save_messages_for_pipeline(messages: list, results_path: str, alg: SecurityAlgorithm, action: ActionType,
                               starting_index: int):
    should_override = is_new_execution(starting_index)
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
    params = extract_arguments()
    last_message_index = get_last_message_index()

    encryption_algorithm_type = convert_int_to_alg_type(params.encryption_algorithm)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               params.cipher_block_mode,
                                                                               params.min_key_value,
                                                                               params.max_key_value)

    extract_key_for_algorithm(params.key_file, encryption_instance, action_type, last_message_index)
    transformed_messages = []

    storage = CheckpointStorage(alg=encryption_instance, results_path=params.path_for_result_messages,
                                transformed_messages=transformed_messages, action_type=action_type,
                                initial_message_index=last_message_index)
    signal.signal(signal.SIGBREAK, partial(handle_signal, storage=storage))
    signal.signal(signal.SIGTERM, partial(handle_signal, storage=storage))

    messages = get_message(params.path_for_messages, encryption_instance, action_type, last_message_index)
    for message in messages:
        transformed_msg = get_transformed_message(message, action_type, encryption_instance)
        transformed_messages.append(transformed_msg)
    save_messages_for_pipeline(transformed_messages, params.path_for_result_messages, encryption_instance, action_type,
                               last_message_index)

    return transformed_messages


def execute_operation(messages: list[int], action: ActionType, algorithm: SecurityAlgorithm, total_checkpoint) -> int:
    if total_checkpoint:
        deserialized_total = algorithm.deserialize_message(total_checkpoint)
    else:
        deserialized_total = None

    if action == ActionType.Addition:
        encrypted_res = algorithm.calc_encrypted_sum(messages, start_total=deserialized_total,
                                                     checkpoint_callback=checkpoint_callback)
    elif action == ActionType.Multiplication:
        encrypted_res = algorithm.calc_encrypted_multiplication(messages, start_total=deserialized_total,
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

    extract_key_for_algorithm(params.key_file, encryption_instance, action_type, last_message_index)
    transformed_messages = []

    checkpoint_storage = OperationCheckpointStorage(
        alg=encryption_instance,
        results_path=params.path_for_result_messages,
        transformed_messages=transformed_messages,
        action_type=action_type,
        initial_message_index=last_message_index
    )

    signal.signal(signal.SIGBREAK, partial(handle_signal, storage=checkpoint_storage))
    signal.signal(signal.SIGTERM, partial(handle_signal, storage=checkpoint_storage))

    messages = get_message(params.path_for_messages, encryption_instance, action_type, last_message_index)

    operation_encrypted_result = execute_operation(messages, action_type, encryption_instance, total_checkpoint)
    print(f"The original messages: {messages}")
    print(f"The decrypted result: {operation_encrypted_result}")
    return operation_encrypted_result
