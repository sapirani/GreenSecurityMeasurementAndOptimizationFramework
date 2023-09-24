# what directories contain measurement results
import os

from Scanner.general_consts import HardwareColumns

# Paths
MEASUREMENTS_DIR_PATH = r"C:\Users\sapir\שולחן העבודה\University\Second Degree\Green Security\Project\Scanner\LENOVO Yoga 7 15ITL5 82BJ Windows 10\No Scan\Power Saver Plan"
IDLE_DIRECTORY_PATH = r"C:\Users\sapir\שולחן העבודה\University\Second Degree\Green Security\Project\Scanner\LENOVO Yoga 7 15ITL5 82BJ Windows 10\No Scan\Power Saver Plan\Measurement 1"

# Summary File related
TOTAL_COL_SUMMARY = "Total"
SUMMARY_FILE_NAME = 'summary.xlsx'

class SummaryFields:
    DURATION = "Duration"
    CPU = "CPU"
    MEMORY = "Memory (MB)"
    PAGE_FAULTS = "Page Faults"
    IO_READ_BYTES = "IO Read (KB - sum)"
    IO_READ_COUNT = "IO Read Count (# - sum)"
    IO_WRITE_BYTES = "IO Write (KB - sum)"
    IO_WRITE_COUNT = "IO Write Count (# - sum)"
    DISK_IO_READ_TIME = "Disk IO Read Time (ms - sum)"
    DISK_IO_WRITE_TIME = "Disk IO Write Time (ms - sum)"
    ENERGY_CONSUMPTION = "Energy consumption - total energy(mwh)"
    TOTAL_COLUMN = "Toal"


# Processes File related
PROCESSES_FILE_NAME = "processes_data.csv"
HEAVYLOAD_PROCESS_NAME = "HeavyLoad"

class AllProcessesFileFields:
    PROCESS_NAME_COL = "PNAME"
    CPU = "CPU(%)"
    MEMORY = "MEMORY(MB)"
    DISK_READ_BYTES = "READ_IO(KB)"
    DISK_READ_COUNT = "READ_IO(#)"
    DISK_WRITE_BYTES = "WRITE_IO(KB)"
    DISK_WRITE_COUNT = "WRITE_IO(#)"
    PAGE_FAULTS = "PAGE_FAULTS"

# Hardware Information File related
HARDWARE_INFORMATION_NAME = 'hardware_information.csv'

# column names in dataset

# *** Process Related Cols *** #
class ProcessColumns:
    CPU_PROCESS_COL = "cpu_usage_process"
    MEMORY_PROCESS_COL = "memory_usage_process"
    DISK_READ_BYTES_PROCESS_COL = "disk_read_bytes_usage_process"
    DISK_READ_COUNT_PROCESS_COL = "disk_read_count_usage_process"
    DISK_WRITE_BYTES_PROCESS_COL = "disk_write_bytes_usage_process"
    DISK_WRITE_COUNT_PROCESS_COL = "disk_write_count_usage_process"
    PAGE_FAULTS_PROCESS_COL = "number_of_page_faults_process"
    ENERGY_USAGE_PROCESS_COL = "energy_consumption_process_mWh"  # total - idle, Target col


# *** System Related Cols *** #
class SystemColumns:
    CPU_SYSTEM_COL = "cpu_usage_system"
    MEMORY_SYSTEM_COL = "memory_usage_system"
    DISK_READ_BYTES_SYSTEM_COL = "disk_read_bytes_usage_system"
    DISK_READ_COUNT_SYSTEM_COL = "disk_read_count_usage_system"
    DISK_WRITE_BYTES_SYSTEM_COL = "disk_write_bytes_usage_system"
    DISK_WRITE_COUNT_SYSTEM_COL = "disk_write_count_usage_system"
    DISK_READ_TIME = "disk_read_time_system_ms_sum"
    DISK_WRITE_TIME = "disk_write_time_system_ms_sum"

    DURATION_COL = "duration"  # ??? maybe give up
    ENERGY_TOTAL_USAGE_SYSTEM_COL = "total_energy_consumption_system_mWh"
    PAGE_FAULT_SYSTEM_COL = "number_of_page_faults_system"


# *** IDLE State Related Cols *** #
class IDLEColumns:
    CPU_IDLE_COL = "cpu_usage_idle"
    MEMORY_IDLE_COL = "memory_usage_idle"
    DISK_READ_BYTES_IDLE_COL = "disk_read_bytes_usage_idle"
    DISK_READ_COUNT_IDLE_COL = "disk_read_count_usage_idle"
    DISK_WRITE_BYTES_IDLE_COL = "disk_write_bytes_usage_idle"
    DISK_WRITE_COUNT_IDLE_COL = "disk_write_count_usage_idle"
    DISK_READ_TIME = "disk_read_time_idle_ms_sum"
    DISK_WRITE_TIME = "disk_write_time_idle_ms_sum"

    ENERGY_TOTAL_USAGE_IDLE_COL = "total_energy_consumption_in_idle_mWh"
    PAGE_FAULT_IDLE_COL = "number_of_page_faults_idle"
    COMPARED_TO_IDLE = "compared_to_idle"
    DURATION_COL = "duration"  # ??


# *** Hardware Related Cols *** #



DATASET_COLUMNS = [ProcessColumns.CPU_PROCESS_COL, ProcessColumns.MEMORY_PROCESS_COL,
                   ProcessColumns.DISK_READ_BYTES_PROCESS_COL, ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                   ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL, ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL, ProcessColumns.PAGE_FAULTS_PROCESS_COL,
                   SystemColumns.DURATION_COL, SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                   SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                   SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                   SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME, SystemColumns.PAGE_FAULT_SYSTEM_COL, SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL,
                   IDLEColumns.DURATION_COL, IDLEColumns.CPU_IDLE_COL, IDLEColumns.MEMORY_IDLE_COL,
                   IDLEColumns.DISK_READ_BYTES_IDLE_COL, IDLEColumns.DISK_READ_COUNT_IDLE_COL,
                   IDLEColumns.DISK_WRITE_BYTES_IDLE_COL, IDLEColumns.DISK_WRITE_COUNT_IDLE_COL,
                   IDLEColumns.DISK_READ_TIME, IDLEColumns.DISK_WRITE_TIME, IDLEColumns.PAGE_FAULT_IDLE_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL,
                   HardwareColumns.PC_TYPE, HardwareColumns.PC_MANUFACTURER, HardwareColumns.SYSTEM_FAMILY, HardwareColumns.MACHINE_TYPE,
                   HardwareColumns.DEVICE_NAME, HardwareColumns.OPERATING_SYSTEM, HardwareColumns.OPERATING_SYSTEM_RELEASE, HardwareColumns.OPERATING_SYSTEM_VERSION,
                   HardwareColumns.PROCESSOR_NAME, HardwareColumns.PROCESSOR_PHYSICAL_CORES, HardwareColumns.PROCESSOR_TOTAL_CORES, HardwareColumns.PROCESSOR_MAX_FREQ,
                   HardwareColumns.PROCESSOR_MIN_FREQ, HardwareColumns.TOTAL_RAM,
                   HardwareColumns.PHYSICAL_DISK_NAME, HardwareColumns.PHYSICAL_DISK_MANUFACTURER, HardwareColumns.PHYSICAL_DISK_MODEL,
                   HardwareColumns.PHYSICAL_DISK_MEDIA_TYPE, HardwareColumns.LOGICAL_DISK_NAME, HardwareColumns.LOGICAL_DISK_MANUFACTURER,
                   HardwareColumns.LOGICAL_DISK_MODEL, HardwareColumns.LOGICAL_DISK_DISK_TYPE, HardwareColumns.LOGICAL_DISK_PARTITION_STYLE,
                   HardwareColumns.LOGICAL_DISK_NUMBER_OF_PARTITIONS, HardwareColumns.PHYSICAL_SECTOR_SIZE, HardwareColumns.LOGICAL_SECTOR_SIZE,
                   HardwareColumns.BUS_TYPE, HardwareColumns.FILESYSTEM, HardwareColumns.BATTERY_DESIGN_CAPACITY, HardwareColumns.FULLY_CHARGED_BATTERY_CAPACITY]

