from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.encryption_algorithm_factory import EncryptionAlgorithmFactory
from tasks.confidential_computing_tasks.encryption_type import EncryptionType
from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.homomorphic_security_algorithm import \
    HomomorphicSecurityAlgorithm
from tasks.confidential_computing_tasks.tasks_options.execute_pipeline import execute_operation_pipeline
from tasks.confidential_computing_tasks.utils.algorithm_utils import extract_arguments, convert_int_to_alg_type
from tasks.confidential_computing_tasks.utils.saving_utils import extract_messages_from_file

# HOMOMORPHIC_ALGORITHMS = [EncryptionType.Paillier, EncryptionType.RSA, EncryptionType.LightPhePaillier,
#                           EncryptionType.LightPheRSA,
#                           EncryptionType.LightPheBenaloh, EncryptionType.LightPheElGamal,
#                           EncryptionType.LightPheExponentialElGamal,
#                           EncryptionType.LightPheEllipticCurveElGamal, EncryptionType.LightPheOkamotoUchiyama,
#                           EncryptionType.LightPheDamgardJurik,
#                           EncryptionType.LightPheNaccacheStern, EncryptionType.LightPheGoldwasserMicali,
#                           EncryptionType.BFVTenseal, EncryptionType.CKKSTenseal]
#
#
# def calc_sum_using_homomorphic_encryption(messages: list[int], algorithm: HomomorphicSecurityAlgorithm) -> int:
#     encrypted_messages = [encryption_instance.encrypt_message(msg) for msg in messages]
#     total_encrypted_sum = encrypted_messages[0]
#     for enc_message in encrypted_messages[1:]:
#         total_encrypted_sum = algorithm.add_messages(total_encrypted_sum, enc_message)
#
#     return total_encrypted_sum
#
#
# def calc_sum_using_regular_encryption(messages: list[int], algorithm: SecurityAlgorithm) -> int:
#     total_sum = sum(messages)
#     return algorithm.encrypt_message(total_sum)


if __name__ == '__main__':
    execute_operation_pipeline(ActionType.Addition)
    # params = extract_arguments()
    #
    # messages_file = params.path_for_messages
    # result_messages_file = params.path_for_result_messages
    # encryption_algorithm = params.encryption_algorithm
    # encryption_key_file = params.key_file
    # min_key_val = params.min_key_value
    # max_key_val = params.max_key_value
    # cipher_block_mode = params.cipher_block_mode
    #
    # encryption_algorithm_type = convert_int_to_alg_type(encryption_algorithm)
    #
    # if encryption_algorithm in HOMOMORPHIC_ALGORITHMS:
    #     encryption_instance = EncryptionAlgorithmFactory.create_homomorphic_algorithm(encryption_algorithm_type,
    #                                                                                   min_key_val,
    #                                                                                   max_key_val)
    # else:
    #     encryption_instance = EncryptionAlgorithmFactory.create_security_algorithm(encryption_algorithm_type,
    #                                                                                cipher_block_mode,
    #                                                                                min_key_val,
    #                                                                                max_key_val)
    #
    # encryption_instance.extract_key(encryption_key_file)
    # messages = extract_messages_from_file(messages_file)
    #
    # if encryption_algorithm in HOMOMORPHIC_ALGORITHMS:
    #     encrypted_sum = calc_sum_using_homomorphic_encryption(messages, encryption_instance)
    # else:
    #     encrypted_sum = calc_sum_using_regular_encryption(messages, encryption_instance)
    #
    # decrypted_sum = encryption_instance.decrypt_message(encrypted_sum)
    # print(f"The original messages: {messages}")
    # print(f"The encrypted sum: {encrypted_sum}")
    # print(f"The decrypted sum: {decrypted_sum}")
