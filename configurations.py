from enum import Enum
import os.path
from pathlib import Path


class ScanOption(Enum):
    NO_SCAN = 1
    ONE_SCAN = 2
    CONTINUOUS_SCAN = 3


MINUTE = 60

# ======= Program Parameters =======
scan_option = ScanOption.CONTINUOUS_SCAN
scan_type = "QuickScan"
MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 1 * MINUTE


# ======= Result Data Paths =======
def calc_dir():
    if scan_option == ScanOption.NO_SCAN:
        return 'no_scan'
    elif scan_option == ScanOption.ONE_SCAN:
        return os.path.join('one_scan', scan_type)
    else:
        return os.path.join('continuous_scan', scan_type)


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
