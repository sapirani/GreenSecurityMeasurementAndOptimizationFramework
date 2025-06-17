from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.algorithm_utils import extract_arguments, convert_str_to_alg_type, \
    extract_messages_from_file, get_updated_message
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory


def execute_pipeline(action_type: ActionType) -> list[int]:
    messages_file, encryption_algorithm, encryption_key_file, min_key_val, max_key_val = extract_arguments()
    encryption_algorithm_type = convert_str_to_alg_type(encryption_algorithm)
    messages = extract_messages_from_file(messages_file)
    encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                               min_key_val,
                                                                               max_key_val)
    encryption_instance.extract_key(encryption_key_file)

    updated_messages = []
    for message in messages:
        updated_msg = get_updated_message(message, action_type, encryption_instance)
        updated_messages.append(updated_msg)

    return updated_messages