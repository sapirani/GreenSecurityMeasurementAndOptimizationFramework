import math

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, convert_int_to_alg_type, \
    get_updated_message
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file, \
    write_messages_to_file

HOMOMORPHIC_ALGORITHMS = [EncryptionType.Paillier, EncryptionType.RSA, EncryptionType.LightPhePaillier,
                          EncryptionType.LightPheRSA,
                          EncryptionType.LightPheBenaloh, EncryptionType.LightPheElGamal,
                          EncryptionType.LightPheExponentialElGamal,
                          EncryptionType.LightPheEllipticCurveElGamal, EncryptionType.LightPheOkamotoUchiyama,
                          EncryptionType.LightPheDamgardJurik,
                          EncryptionType.LightPheNaccacheStern, EncryptionType.LightPheGoldwasserMicali,
                          EncryptionType.BFVTenseal, EncryptionType.CKKSTenseal]


def calc_sum_using_homomorphic_encryption(messages: list[int], algorithm: HomomorphicSecurityAlgorithm) -> int:
    encrypted_messages = [algorithm.encrypt_message(msg) for msg in messages]
    total_encrypted_sum = encrypted_messages[0]
    for enc_message in encrypted_messages[1:]:
        total_encrypted_sum = algorithm.add_messages(total_encrypted_sum, enc_message)

    return total_encrypted_sum


def calc_sum_using_regular_encryption(messages: list[int], algorithm: SecurityAlgorithm) -> int:
    total_sum = sum(messages)
    return algorithm.encrypt_message(total_sum)

def calc_mul_using_homomorphic_encryption(messages: list[int], algorithm: HomomorphicSecurityAlgorithm) -> int:
    encrypted_messages = [algorithm.encrypt_message(msg) for msg in messages]
    total_encrypted_mul = encrypted_messages[0]
    for enc_message in encrypted_messages[1:]:
        total_encrypted_mul = algorithm.multiply_messages(total_encrypted_mul, enc_message)

    return total_encrypted_mul


def calc_mul_using_regular_encryption(messages: list[int], algorithm: SecurityAlgorithm) -> int:
    total_mul = math.prod(messages)
    return algorithm.encrypt_message(total_mul)


def execute_regular_pipeline(action_type: ActionType) -> list[int]:
    params = extract_arguments()

    messages_file = params.path_for_messages
    result_messages_file = params.path_for_result_messages
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

    if action_type == ActionType.Encryption or action_type == ActionType.FullPipeline:
        messages = extract_messages_from_file(messages_file)
    elif action_type == ActionType.Decryption:
        messages = encryption_instance.load_encrypted_messages(messages_file)
    else:
        messages = []

    updated_messages = []
    for message in messages:
        updated_msg = get_updated_message(message, action_type, encryption_instance)
        updated_messages.append(updated_msg)

    if action_type == ActionType.Encryption:
        encryption_instance.save_encrypted_messages(updated_messages, result_messages_file)
    # If decryption or full pipeline, optionally save decrypted ints as text
    elif action_type in (ActionType.Decryption, ActionType.FullPipeline):
        write_messages_to_file(result_messages_file, updated_messages)
    else:
        raise Exception("Unknown action type.")

    return updated_messages

def execute_operation(messages: list[int], action: ActionType, algorithm: SecurityAlgorithm, encryption_algorithm: EncryptionType) -> int:
    if encryption_algorithm in HOMOMORPHIC_ALGORITHMS:
        if action == ActionType.Addition:
            return calc_sum_using_homomorphic_encryption(messages, algorithm)
        elif action == ActionType.Multiplication:
            return calc_mul_using_homomorphic_encryption(messages, algorithm)
    else:
        if action == ActionType.Addition:
            return calc_sum_using_regular_encryption(messages, algorithm)
        elif action == ActionType.Multiplication:
            return calc_mul_using_regular_encryption(messages, algorithm)

    raise Exception("Unknown encryption algorithm or action type.")

def execute_operation_pipeline(action_type: ActionType) -> int:
    params = extract_arguments()
    messages_file = params.path_for_messages
    encryption_algorithm = params.encryption_algorithm
    encryption_key_file = params.key_file
    min_key_val = params.min_key_value
    max_key_val = params.max_key_value
    cipher_block_mode = params.cipher_block_mode

    encryption_algorithm_type = convert_int_to_alg_type(encryption_algorithm)

    if encryption_algorithm in HOMOMORPHIC_ALGORITHMS:
        encryption_instance = EncryptionAlgorithmFactory.create_homomorphic_algorithm(encryption_algorithm_type,
                                                                                      min_key_val,
                                                                                      max_key_val)
    else:
        encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
                                                                                   cipher_block_mode,
                                                                                   min_key_val,
                                                                                   max_key_val)

    encryption_instance.extract_key(encryption_key_file)
    messages = extract_messages_from_file(messages_file)

    operation_encrypted_result = execute_operation(messages, action_type, encryption_instance, encryption_algorithm_type)
    print(f"The original messages: {messages}")
    print(f"The decrypted result: {operation_encrypted_result}")
    return operation_encrypted_result