from enum import Enum
import os.path
from pathlib import Path


class ScanOption(Enum):
    NO_SCAN = 1
    ONE_SCAN = 2
    CONTINUOUS_SCAN = 3


class PowerPlans:
    BALANCED = ("Balanced Plan", "381b4222-f694-41f0-9685-ff5bb260df2e")
    POWER_SAVER = ("Power Saver Plan", "a1841308-3541-4fab-bc81-f71556f20b4a")
    HIGH_PERFORMANCE = ("High Performance Plan", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")


MINUTE = 60

# ======= Program Parameters =======
power_plan = PowerPlans.BALANCED
scan_option = ScanOption.CONTINUOUS_SCAN
scan_type = "QuickScan"
MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 1 * MINUTE


# ======= Power Plan Name and GUID (do not change) =======
power_plan_name = power_plan[0]
power_plan_guid = power_plan[1]


# ======= Result Data Paths =======
def calc_dir():
    if scan_option == ScanOption.NO_SCAN:
        return os.path.join(power_plan_name, 'No Scan', scan_type)
    elif scan_option == ScanOption.ONE_SCAN:
        return os.path.join(power_plan_name, 'One Scan', scan_type)
    else:
        return os.path.join(power_plan_name, 'Continuous Scan', scan_type)


results_dir = calc_dir()
GRAPHS_DIR = os.path.join(results_dir, "graphs")

Path(GRAPHS_DIR).mkdir(parents=True, exist_ok=True)

PROCESSES_CSV = os.path.join(results_dir, 'processes_data.csv')
TOTAL_MEMORY_EACH_MOMENT_CSV = os.path.join(results_dir, 'total_memory_each_moment.csv')
DISK_IO_EACH_MOMENT = os.path.join(results_dir, 'disk_io_each_moment.csv')
BATTERY_STATUS_CSV = os.path.join(results_dir, 'battery_status.csv')
GENERAL_INFORMATION_FILE = os.path.join(results_dir, 'general_information.txt')
TOTAL_CPU_CSV = os.path.join(results_dir, 'total_cpu.csv')


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


# ======= Constants =======
GB = 2**30
MB = 2**20
KB = 2**10
ANTIVIRUS_PROCESS_NAME = "MsMpeng"

