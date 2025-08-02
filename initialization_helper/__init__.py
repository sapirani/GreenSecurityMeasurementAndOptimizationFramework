import os.path

from initialization_helper.initialization_factories import running_os_factory, process_resource_usage_recorder_factory, \
    summary_builder_factory, program_to_scan_factory, battery_usage_recorder_factory
from program_parameters import *

# ======= Get Operating System Type =======
running_os = running_os_factory(is_inside_container=is_inside_container)
program = program_to_scan_factory(main_program_to_scan)
background_programs = [program_to_scan_factory(background_program) for background_program in background_programs_types]

processes_resource_usage_recorder = process_resource_usage_recorder_factory(
    process_monitor_type, running_os, program.process_ignore_cond
)
battery_usage_recorder = battery_usage_recorder_factory(battery_monitor_type, running_os)

summary_builder = summary_builder_factory(summary_type)


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


if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT and len(background_programs_types) != 0:
    raise Exception("BASELINE_MEASUREMENT mode can't include background programs!")


def construct_base_dir_path():
    """
    :return: A string represents the directory hierarchy of the results
    """
    computer_info = running_os.get_computer_info(is_inside_container)

    if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name)
    elif scan_option == ScanMode.ONE_SCAN:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'One Scan', program.path_adjustments())
    else:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'Continuous Scan', program.path_adjustments())


base_dir = os.path.abspath(construct_base_dir_path())


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
if (scan_option == ScanMode.CONTINUOUS_SCAN or main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT) and RUNNING_TIME is None:
    raise Exception("MAXIMUM_SCAN_TIME is allowed to be None  only when performing running a regular main program"
                    " in ONE_SCAN mode - the meaning of None is to wait until the main process ends")


if is_inside_container and battery_monitor_type == BatteryMonitorType.FULL:
    raise Exception("Measurement of energy consumption inside container is not supported")
