from program_parameters import messages_to_encrypt_file, security_algorithm_type, \
    algorithm_key_file, min_key_value, results_file_for_encryption, block_cipher_mode, max_key_value, \
    messages_to_decrypt_file, results_file_for_decryption
from tasks.program_classes.baseline_measurement_program import BaselineMeasurementProgram
from tasks.program_classes.server_program import PythonServer
from utils.general_consts import ProgramToScan
from tasks.program_classes.abstract_program import ProgramInterface
from tasks.program_classes.confidential_computing.encryption_pipeline_executor import EncryptionPipelineExecutor
from tasks.program_classes.confidential_computing.message_adder import MessageAdder
from tasks.program_classes.confidential_computing.message_decryptor import MessageDecryptor
from tasks.program_classes.confidential_computing.message_encryptor import MessageEncryptor
from tasks.program_classes.confidential_computing.message_multiplier import MessageMultiplier


def program_to_scan_factory(program_type: ProgramToScan) -> ProgramInterface:
    """
    Return the class that represents the program that the user wishes to run and send its dedicated parameters
    :param program_type: The program specified by the user
    :return: The dedicated class
    """
    if program_type == ProgramToScan.BASELINE_MEASUREMENT:
        return BaselineMeasurementProgram()
    if program_type == ProgramToScan.PythonServer:
        return PythonServer()
    if program_type == ProgramToScan.MessageEncryptor:
        return MessageEncryptor(messages_file=messages_to_encrypt_file, results_file=results_file_for_encryption,
                                security_algorithm=security_algorithm_type, block_mode=block_cipher_mode,
                                key_file=algorithm_key_file, min_key_value=min_key_value, max_key_value=max_key_value)
    if program_type == ProgramToScan.MessageDecryptor:
        return MessageDecryptor(messages_file=messages_to_decrypt_file, results_file=results_file_for_decryption,
                                security_algorithm=security_algorithm_type, block_mode=block_cipher_mode,
                                key_file=algorithm_key_file, min_key_value=min_key_value, max_key_value=max_key_value)
    if program_type == ProgramToScan.MessageAddition:
        return MessageAdder(messages_file=messages_to_encrypt_file, results_file=results_file_for_decryption,
                            security_algorithm=security_algorithm_type, block_mode=block_cipher_mode,
                            key_file=algorithm_key_file, min_key_value=min_key_value, max_key_value=max_key_value)
    if program_type == ProgramToScan.MessageMultiplication:
        return MessageMultiplier(messages_file=messages_to_encrypt_file, results_file=results_file_for_decryption,
                                 security_algorithm=security_algorithm_type, block_mode=block_cipher_mode,
                                 key_file=algorithm_key_file, min_key_value=min_key_value, max_key_value=max_key_value)
    if program_type == ProgramToScan.EncryptionPipelineExecutor:
        return EncryptionPipelineExecutor(messages_file=messages_to_encrypt_file,
                                          results_file=results_file_for_decryption,
                                          security_algorithm=security_algorithm_type, block_mode=block_cipher_mode,
                                          key_file=algorithm_key_file, min_key_value=min_key_value,
                                          max_key_value=max_key_value)

    raise Exception("Choose program to scan from ProgramToScan enum")
