import os.path
from program_parameters import *
import wmi
import platform
from program_classes import *

# ======= Power Plan Name and GUID (do not change) =======
chosen_power_plan_name = power_plan[0]
chosen_power_plan_guid = power_plan[1]

balanced_power_plan_name = PowerPlan.BALANCED[0]
balanced_power_plan_guid = PowerPlan.BALANCED[1]


# ======= Result Data Paths =======
def program_to_scan_factory():
    if program_to_scan == ProgramToScan.ANTIVIRUS:
        return AntivirusProgram(scan_type, custom_scan_path)
    if program_to_scan == ProgramToScan.IDS:
        return IDSProgram(ids_type, interface_name, log_dir)
    if program_to_scan == ProgramToScan.DummyANTIVIRUS:
        return DummyAntivirusProgram(custom_scan_path)

    raise Exception("choose program to scan from ProgramToScan enum")


program = program_to_scan_factory()


def calc_base_dir():
    c = wmi.WMI()
    wmi_system = c.Win32_ComputerSystem()[0]

    computer_info = f"{wmi_system.Manufacturer} {wmi_system.SystemFamily} {wmi_system.Model} " \
                    f"{platform.system()} {platform.release()}"

    if program_to_scan == ProgramToScan.NO_SCAN:
        return os.path.join(computer_info, 'No Scan', chosen_power_plan_name)
    elif scan_option == ScanMode.ONE_SCAN:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'One Scan', program.path_adjustments())
    else:
        return os.path.join(computer_info, program.get_program_name(), chosen_power_plan_name, 'Continuous Scan', program.path_adjustments())


base_dir = calc_base_dir()


def calc_measurement_number(is_scanner=True):
    if measurement_number != NEW_MEASUREMENT:
        return measurement_number

    if not os.path.exists(base_dir):
        return 1

    max_number = max(map(lambda dir_name: int(dir_name[len(MEASUREMENT_NAME_DIR) + 1:]), os.listdir(base_dir)))
    return max_number + 1 if is_scanner else max_number


def result_paths(is_scanner=True):
    measurements_dir = os.path.join(base_dir, f"{MEASUREMENT_NAME_DIR} {calc_measurement_number(is_scanner)}")
    graphs_dir = os.path.join(measurements_dir, "graphs")

    processes_csv = os.path.join(measurements_dir, 'processes_data.csv')
    total_memory_each_moment_csv = os.path.join(measurements_dir, 'total_memory_each_moment.csv')
    disk_io_each_moment = os.path.join(measurements_dir, 'disk_io_each_moment.csv')
    battery_status_csv = os.path.join(measurements_dir, 'battery_status.csv')
    general_information_file = os.path.join(measurements_dir, 'general_information.txt')
    total_cpu_csv = os.path.join(measurements_dir, 'total_cpu.csv')
    summary_csv = os.path.join(measurements_dir, 'summary.csv')
    return measurements_dir, graphs_dir, processes_csv, total_memory_each_moment_csv, disk_io_each_moment,\
        battery_status_csv, general_information_file, total_cpu_csv, summary_csv


# ======= Custom Scan Query (do not change) =======
if program_to_scan == ProgramToScan.ANTIVIRUS and scan_type != ScanType.CUSTOM_SCAN and custom_scan_path != '""':
    raise Exception("custom_scan_path must be empty when running scans other than custom scan")


battery_columns_list = [BatteryColumns.TIME, BatteryColumns.PERCENTS, BatteryColumns.CAPACITY, BatteryColumns.VOLTAGE]

memory_columns_list = [MemoryColumns.TIME, MemoryColumns.USED_MEMORY, MemoryColumns.USED_PERCENT]

cores_names_list = [get_core_name(i) for i in range(1, NUMBER_OF_CORES + 1)]

cpu_columns_list = [CPUColumns.TIME, CPUColumns.USED_PERCENT] + cores_names_list

disk_io_columns_list = [DiskIOColumns.TIME, DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT,
                        DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES, DiskIOColumns.READ_TIME,
                        DiskIOColumns.WRITE_TIME]

processes_columns_list = [
    ProcessesColumns.TIME, ProcessesColumns.PROCESS_ID, ProcessesColumns.PROCESS_NAME, ProcessesColumns.CPU_CONSUMPTION,
    ProcessesColumns.NUMBER_OF_THREADS, ProcessesColumns.USED_MEMORY, ProcessesColumns.MEMORY_PERCENT,
    ProcessesColumns.READ_COUNT, ProcessesColumns.WRITE_COUNT, ProcessesColumns.READ_BYTES, ProcessesColumns.WRITE_BYTES
]
