from program_parameters import antivirus_type, scan_type, custom_scan_path, recursive, should_optimize, \
    should_mitigate_timestomping, ids_type, interface_name, pcap_list_dirs, log_path, configuration_file_path, \
    model_name, model_action, script_relative_path, installation_dir, cpu_percent_to_consume, RUNNING_TIME, \
    memory_chunk_size, consumption_speed, time_interval, messages_to_encrypt_file, security_algorithm_type, \
    algorithm_key_file, min_key_value, results_file_for_encryption, block_cipher_mode, max_key_value, \
    messages_to_decrypt_file, results_file_for_decryption
from general_consts import ProgramToScan, AntivirusType, IDSType
from tasks.program_classes.abstract_program import ProgramInterface
from tasks.program_classes.antiviruses.clam_av_program import ClamAVProgram
from tasks.program_classes.antiviruses.defender_program import DefenderProgram
from tasks.program_classes.antiviruses.dummy_antivirus_program import DummyAntivirusProgram
from tasks.program_classes.antiviruses.sophos_av_program import SophosAVProgram
from tasks.program_classes.confidential_computing.encryption_pipeline_executor import EncryptionPipelineExecutor
from tasks.program_classes.confidential_computing.message_adder import MessageAdder
from tasks.program_classes.confidential_computing.message_decryptor import MessageDecryptor
from tasks.program_classes.confidential_computing.message_encryptor import MessageEncryptor
from tasks.program_classes.confidential_computing.message_multiplier import MessageMultiplier
from tasks.program_classes.dummy_cpu_consumer_program import CPUConsumer
from tasks.program_classes.dummy_io_writer_consumer_program import IOWriteConsumer
from tasks.program_classes.dummy_memory_consumer_program import MemoryConsumer
from tasks.program_classes.ids.snort_program import SnortProgram
from tasks.program_classes.ids.suricata_program import SuricataProgram
from tasks.program_classes.log_anomaly_detection_program import LogAnomalyDetection
from tasks.program_classes.network_receiver_program import NetworkReceiver
from tasks.program_classes.network_sender_program import NetworkSender
from tasks.program_classes.no_scan_program import NoScanProgram
from tasks.program_classes.perfmon_monitoring_program import PerfmonProgram
from tasks.program_classes.server_program import PythonServer
from tasks.program_classes.splunk_program import SplunkProgram
from tasks.program_classes.user_activity_program import UserActivityProgram


def program_to_scan_factory(program_type: ProgramToScan) -> ProgramInterface:
    """
    Return the class that represents the program that the user wishes to run and send its dedicated parameters
    :param program_type: The program specified by the user
    :return: The dedicated class
    """

    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.DEFENDER:
        return DefenderProgram(scan_type, custom_scan_path)
    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.ClamAV:
        return ClamAVProgram(scan_type, custom_scan_path, recursive, should_optimize, should_mitigate_timestomping)
    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.SOPHOS:
        return SophosAVProgram(scan_type, custom_scan_path)
    if program_type == ProgramToScan.IDS and ids_type == IDSType.SURICATA:
        return SuricataProgram(interface_name, pcap_list_dirs, log_path)
    if program_type == ProgramToScan.IDS and ids_type == IDSType.SNORT:
        return SnortProgram(interface_name, pcap_list_dirs, log_path, configuration_file_path=configuration_file_path)
    if program_type == ProgramToScan.DummyANTIVIRUS:
        return DummyAntivirusProgram(custom_scan_path)
    if program_type == ProgramToScan.NO_SCAN:
        return NoScanProgram()
    if program_type == ProgramToScan.Perfmon:
        return PerfmonProgram()
    if program_type == ProgramToScan.UserActivity:
        return UserActivityProgram()
    if program_type == ProgramToScan.LogAnomalyDetection:
        return LogAnomalyDetection(model_name, model_action, script_relative_path, installation_dir)
    if program_type == ProgramToScan.Splunk:
        return SplunkProgram()
    if program_type == ProgramToScan.CPUConsumer:
        return CPUConsumer(cpu_percent_to_consume, RUNNING_TIME)
    if program_type == ProgramToScan.MemoryConsumer:
        return MemoryConsumer(memory_chunk_size, consumption_speed, RUNNING_TIME)
    if program_type == ProgramToScan.IOWriteConsumer:
        return IOWriteConsumer(custom_scan_path)
    if program_type == ProgramToScan.PythonServer:
        return PythonServer()
    if program_type == ProgramToScan.NetworkReceiver:
        return NetworkReceiver()
    if program_type == ProgramToScan.NetworkSender:
        return NetworkSender(time_interval=time_interval, running_time=RUNNING_TIME)
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


    raise Exception("choose program to scan from ProgramToScan enum")
