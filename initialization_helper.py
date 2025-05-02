import os.path

import SummaryVersionImp
from program_parameters import *
from program_classes import *
from os_funcs import *
from scanner_versions_imp import FullScanner, LiteScanner, WithoutBatteryScanner

# ======= Get Operating System Type =======
running_os = None
if platform.system() == "Linux":
    running_os = LinuxOS()
elif platform.system() == "Windows":
    running_os = WindowsOS()

# ====== Get Scanner Version ======
if scanner_version == ScannerVersion.FULL:
    scanner_imp = FullScanner(running_os)
elif scanner_version == ScannerVersion.LITE:
    scanner_imp = LiteScanner(running_os)
elif scanner_version == ScannerVersion.WITHOUT_BATTERY:
    scanner_imp = WithoutBatteryScanner(running_os)

# ====== Get Summary Version ======
if summary_version == SummaryVersion.DUDU:
    summary_version_imp = SummaryVersionImp.DuduSummary()
elif summary_version == SummaryVersion.OTHER:
    summary_version_imp = SummaryVersionImp.OtherSummary()

# ======= Power Plan Name and GUID (do not change) =======
chosen_power_plan_name = power_plan[0]
chosen_power_plan_guid = power_plan[1]
chosen_power_plan_linux_identifier = power_plan[2]

# default plan in Windows
balanced_power_plan_name = PowerPlan.BALANCED[0]
balanced_power_plan_guid = PowerPlan.BALANCED[1]

# default plan in Linux
power_save_plan_name = PowerPlan.POWER_SAVER[0]
power_save_plan_identifier = PowerPlan.POWER_SAVER[2]


# ======= Result Data Paths =======
def program_to_scan_factory(program_type):
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
        return PerfmonProgram(program.get_process_name())
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

    raise Exception("choose program to scan from ProgramToScan enum")


if main_program_to_scan == ProgramToScan.NO_SCAN and len(background_programs_types) != 0:
    raise Exception("NO SCAN mode can't include background programs!")

program = program_to_scan_factory(main_program_to_scan)
background_programs = [program_to_scan_factory(background_program) for background_program in background_programs_types]


def calc_base_dir():
    """
    :return: A string represents the directory hierarchy of the results
    """
    computer_info = running_os.get_computer_info(is_inside_container)

    if main_program_to_scan == ProgramToScan.NO_SCAN:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name)
    elif scan_option == ScanMode.ONE_SCAN:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'One Scan', program.path_adjustments())
    else:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'Continuous Scan', program.path_adjustments())


base_dir = calc_base_dir()


def calc_measurement_number(is_scanner=True):
    """
    return the next number of measurement directory.
    :param is_scanner: True if we are running the scanner. False if we  are running analyzer
    :return:  one of the followings:
    1.  The number specified by the user.
    2.  The highest dir number exists + 1 if the user chose NEW_MEASUREMENT and runs the scanner.
    2.  The highest dir number exists if the user chose NEW_MEASUREMENT and runs the analyzer.

    """
    if measurement_number != NEW_MEASUREMENT:
        return measurement_number

    if not os.path.exists(base_dir):
        return 1

    max_number = max(map(lambda dir_name: int(dir_name[len(MEASUREMENT_NAME_DIR) + 1:]), os.listdir(base_dir)))
    return max_number + 1 if is_scanner else max_number


def result_paths(is_scanner=True):
    """
    return all paths of results files
    :param is_scanner:
    :return:
    """
    measurements_dir = os.path.join(base_dir, f"{MEASUREMENT_NAME_DIR} {calc_measurement_number(is_scanner)}")
    graphs_dir = os.path.join(measurements_dir, "graphs")
    stdout_files_dir = os.path.join(measurements_dir, "stdouts")
    stderr_files_dir = os.path.join(measurements_dir, "stderrs")

    processes_csv = os.path.join(measurements_dir, 'processes_data.csv')
    total_memory_each_moment_csv = os.path.join(measurements_dir, 'total_memory_each_moment.csv')
    disk_io_each_moment = os.path.join(measurements_dir, 'disk_io_each_moment.csv')
    network_io_each_moment = os.path.join(measurements_dir, 'network_io_each_moment.csv')
    battery_status_csv = os.path.join(measurements_dir, 'battery_status.csv')
    general_information_file = os.path.join(measurements_dir, 'general_information.txt')
    total_cpu_csv = os.path.join(measurements_dir, 'total_cpu.csv')
    summary_csv = os.path.join(measurements_dir, 'summary.xlsx')

    program.set_results_dir(measurements_dir)
    for background_program in background_programs:
        background_program.set_results_dir(measurements_dir)

    return measurements_dir, graphs_dir, stdout_files_dir, stderr_files_dir, processes_csv, total_memory_each_moment_csv, \
        disk_io_each_moment, network_io_each_moment, battery_status_csv, general_information_file, total_cpu_csv, \
        summary_csv


# ======= Custom Scan Query (do not change) =======
if main_program_to_scan == ProgramToScan.ANTIVIRUS and ProgramToScan.DummyANTIVIRUS not in background_programs_types \
        and scan_type != ScanType.CUSTOM_SCAN and custom_scan_path != '""':
    raise Exception("custom_scan_path must be empty when running scans other than custom scan")

# ======= IDS Checks =======
if (pcap_list_dirs is None or len(pcap_list_dirs) == 0) and interface_name is None:
    raise Exception("Choose interface for IDS to listen on or provide pcap directories list to analyse")

if (pcap_list_dirs is not None and len(pcap_list_dirs) > 0) and interface_name is not None:
    raise Exception("Choose either interface to listen on or pcap files when using IDS, not both")

# ======= Scan Time Checks =======
if (scan_option == ScanMode.CONTINUOUS_SCAN or main_program_to_scan == ProgramToScan.NO_SCAN) and RUNNING_TIME is None:
    raise Exception("MAXIMUM_SCAN_TIME is allowed to be None  only when performing running a regular main program"
                    " in ONE_SCAN mode - the meaning of None is to wait until the main process ends")


if is_inside_container and scanner_version == ScannerVersion.FULL:
    raise Exception("Measurement of energy consumption inside container is not supported")

# ======= Prepare dataframes titles =======
battery_columns_list = [BatteryColumns.TIME, BatteryColumns.PERCENTS, BatteryColumns.CAPACITY, BatteryColumns.VOLTAGE]

memory_columns_list = [MemoryColumns.TIME, MemoryColumns.USED_MEMORY, MemoryColumns.USED_PERCENT]

cores_names_list = [get_core_name(i) for i in range(1, NUMBER_OF_CORES + 1)]

cpu_columns_list = [CPUColumns.TIME, CPUColumns.USED_PERCENT] + cores_names_list

disk_io_columns_list = [DiskIOColumns.TIME, DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT,
                        DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES, DiskIOColumns.READ_TIME,
                        DiskIOColumns.WRITE_TIME]

network_io_columns_list = [NetworkIOColumns.TIME, NetworkIOColumns.PACKETS_SENT, NetworkIOColumns.PACKETS_RECEIVED,
                           NetworkIOColumns.KB_SENT, NetworkIOColumns.KB_RECEIVED]

processes_columns_list = [
    ProcessesColumns.TIME, ProcessesColumns.PROCESS_ID, ProcessesColumns.PROCESS_NAME, ProcessesColumns.CPU_CONSUMPTION,
    ProcessesColumns.NUMBER_OF_THREADS, ProcessesColumns.USED_MEMORY, ProcessesColumns.MEMORY_PERCENT,
    ProcessesColumns.READ_COUNT, ProcessesColumns.WRITE_COUNT, ProcessesColumns.READ_BYTES, ProcessesColumns.WRITE_BYTES,
    ProcessesColumns.PAGE_FAULTS,
    ProcessesColumns.BYTES_SENT, ProcessesColumns.PACKETS_SENT,
    ProcessesColumns.BYTES_RECEIVED, ProcessesColumns.PACKETS_RECEIVED
]
