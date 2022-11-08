from enum import Enum
import os.path


# ======= Constants =======
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
NEW_MEASUREMENT = -1
MEASUREMENT_NAME_DIR = "Measurement"


class ScanMode(Enum):
    NO_SCAN = 1
    ONE_SCAN = 2
    CONTINUOUS_SCAN = 3


class PowerPlan:
    BALANCED = ("Balanced Plan", "381b4222-f694-41f0-9685-ff5bb260df2e")
    POWER_SAVER = ("Power Saver Plan", "a1841308-3541-4fab-bc81-f71556f20b4a")
    HIGH_PERFORMANCE = ("High Performance Plan", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")


class ScanType:
    FULL_SCAN = "FullScan"
    QUICK_SCAN = "QuickScan"
    CUSTOM_SCAN = "CustomScan"


MINUTE = 60

# ======= Scanner Parameters =======
power_plan = PowerPlan.HIGH_PERFORMANCE
scan_option = ScanMode.NO_SCAN
scan_type = ScanType.FULL_SCAN  # relevant only for one scan or continuous scan
custom_scan_path = r""  # relevant only for custom scans. On other types, must be empty
MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 0.5 * MINUTE

measurement_number = 2000    # write number between 1->inf or type NEW_MEASUREMENT

# MUST disable tamper protection manually for this feature to work
disable_real_time_protection_during_measurement = True  # must use administrator permissions

screen_brightness_level = 75    # A number between 0 and 100

# return to default settings (can be costumed)
DEFAULT_SCREEN_TURNS_OFF_TIME = 4
DEFAULT_TIME_BEFORE_SLEEP_MODE = 4

# ======= Power Plan Name and GUID (do not change) =======
chosen_power_plan_name = power_plan[0]
chosen_power_plan_guid = power_plan[1]

balanced_power_plan_name = PowerPlan.BALANCED[0]
balanced_power_plan_guid = PowerPlan.BALANCED[1]

# ======= Custom Scan Query (do not change) =======
custom_scan_query = ""
if scan_type != ScanType.CUSTOM_SCAN and custom_scan_path != "":
    raise Exception("scan_type must be empty when running scans other than custom scan")

if scan_type == ScanType.CUSTOM_SCAN:
    custom_scan_query = f" -ScanPath {custom_scan_path}"


# ======= Result Data Paths =======
def calc_base_dir():
    if scan_option == ScanMode.NO_SCAN:
        return os.path.join(chosen_power_plan_name, 'No Scan')
    elif scan_option == ScanMode.ONE_SCAN:
        return os.path.join(chosen_power_plan_name, 'One Scan', scan_type)
    else:
        return os.path.join(chosen_power_plan_name, 'Continuous Scan', scan_type)


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
    return measurements_dir, graphs_dir, processes_csv, total_memory_each_moment_csv, disk_io_each_moment, battery_status_csv, \
           general_information_file, total_cpu_csv


# ======= Table Column Names =======
class BatteryColumns:
    TIME = "Time(sec)"
    PERCENTS = "REMAINING BATTERY(%)"
    CAPACITY = "REMAINING CAPACITY(mWh)"
    VOLTAGE = "Voltage(mV)"


battery_columns_list = [BatteryColumns.TIME, BatteryColumns.PERCENTS, BatteryColumns.CAPACITY, BatteryColumns.VOLTAGE]


class MemoryColumns:
    TIME = "Time(sec)"
    USED_MEMORY = "Used(GB)"
    USED_PERCENT = "Percentage"


memory_columns_list = [MemoryColumns.TIME, MemoryColumns.USED_MEMORY, MemoryColumns.USED_PERCENT]


class CPUColumns:
    TIME = "Time(sec)"
    USED_PERCENT = "Percentage"


cpu_columns_list = [CPUColumns.TIME, CPUColumns.USED_PERCENT]


class DiskIOColumns:
    TIME = "Time(sec)"
    READ_COUNT = "READ(#)"
    WRITE_COUNT = "WRITE(#)"
    READ_BYTES = "READ(KB)"
    WRITE_BYTES = "WRITE(KB)"


disk_io_columns_list = [DiskIOColumns.TIME, DiskIOColumns.READ_COUNT, DiskIOColumns.WRITE_COUNT,
                        DiskIOColumns.READ_BYTES, DiskIOColumns.WRITE_BYTES]


class ProcessesColumns:
    TIME = "Time(sec)"
    PROCESS_ID = "PID"
    PROCESS_NAME = "PNAME"
    CPU_CONSUMPTION = "CPU(%)"
    NUMBER_OF_THREADS = "NUM THREADS"
    USED_MEMORY = "MEMORY(MB)"
    MEMORY_PERCENT = "MEMORY(%)"
    READ_COUNT = "READ_IO(#)"
    WRITE_COUNT = "WRITE_IO(#)"
    READ_BYTES = "READ_IO(KB)"
    WRITE_BYTES = "WRITE_IO(KB)"


processes_columns_list = [
    ProcessesColumns.TIME, ProcessesColumns.PROCESS_ID, ProcessesColumns.PROCESS_NAME, ProcessesColumns.CPU_CONSUMPTION,
    ProcessesColumns.NUMBER_OF_THREADS, ProcessesColumns.USED_MEMORY, ProcessesColumns.MEMORY_PERCENT,
    ProcessesColumns.READ_COUNT, ProcessesColumns.WRITE_COUNT, ProcessesColumns.READ_BYTES, ProcessesColumns.WRITE_BYTES
]
