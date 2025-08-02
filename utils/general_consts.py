from enum import Enum
import psutil

# ======= Constants =======
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
NEW_MEASUREMENT = -1
MEASUREMENT_NAME_DIR = "Measurement"
PROCESS_ID_PHRASE = "Main Process ID"
BACKGROUND_ID_PHRASE = "Background Processes IDs"
NUMBER_OF_CORES = psutil.cpu_count()

SECOND = 1
MINUTE = 60

NEVER_TURN_SCREEN_OFF = 0
NEVER_GO_TO_SLEEP_MODE = 0

YES_BUTTON = 6
NO_BUTTON = 7

# ======= Constants =======
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
SYSTEM_IDLE_PID = 0

pc_types = ["Unspecified", "Desktop", "Mobile Device", "Workstation",
                "EnterpriseServer", "SOHOServer", "AppliancePC", "PerformanceServer"]

physical_memory_types = ['Invalid', 'Other', 'Unknown', 'DRAM', # 00-03h
                         'EDRAM', 'VRAM', 'SRAM', 'RAM', # 04-07h
                         'ROM', 'FLASH', 'EEPROM', 'FEPROM', # 08-0Bh
                         'EPROM', 'CDRAM', '3DRAM', 'SDRAM', # 0C-0Fh
                         'SGRAM', 'RDRAM', 'DDR', 'DDR2', # 10-13h
                         'DDR2 FB-DIMM', 'Reserved', 'Reserved', 'Reserved', # 14-17h
                         'DDR3', 'FBD2', 'DDR4', 'LPDDR', # 18-1Bh
                         'LPDDR2', 'LPDDR3', 'LPDDR4', 'Logical non-volatile device' # 1C-1Fh
                         'HBM (High Bandwidth Memory)', 'HBM2 (High Bandwidth Memory Generation 2)',
                         'DDR5', 'LPDDR5' # 20-23h
                         ]

disk_types = ["Unknown", "NoRootDirectory", "Removable", "Fixed", "Network", "CDRom", "RAM disk"]


# ======= Static Classes =======
class ProgramToScan(Enum):
    BASELINE_MEASUREMENT = 1
    ANTIVIRUS = 2
    DummyANTIVIRUS = 3
    IDS = 4
    Perfmon = 5
    UserActivity = 6
    LogAnomalyDetection = 7
    Splunk = 8
    CPUConsumer = 9
    MemoryConsumer = 10
    IOWriteConsumer = 11
    PythonServer = 12
    NetworkReceiver = 13
    NetworkSender = 14


class ScanMode(Enum):
    ONE_SCAN = 1
    CONTINUOUS_SCAN = 2


class BatteryMonitorType(Enum):
    FULL = 1
    WITHOUT_BATTERY = 2


class ProcessMonitorType(Enum):
    FULL = 1
    PROCESSES_OF_INTEREST_ONLY = 2


class SummaryType(Enum):
    ISOLATE_SYSTEM_RESOURCES = 1
    NATIVE = 2


class PowerPlan:
    BALANCED = ("Balanced Plan", "381b4222-f694-41f0-9685-ff5bb260df2e", None)
    POWER_SAVER = ("Power Saver Plan", "a1841308-3541-4fab-bc81-f71556f20b4a", "powersave")
    HIGH_PERFORMANCE = ("High Performance Plan", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", "performance")


class ScanType:
    FULL_SCAN = "FullScan"
    QUICK_SCAN = "QuickScan"
    CUSTOM_SCAN = "CustomScan"
    
    
class IDSType:
    SURICATA = "Suricata"
    SNORT = "Snort"


class AntivirusType:
    DEFENDER = "Defender"
    ClamAV = "ClamAV"
    SOPHOS = "Sophos"


class TableNames:
    CPU = "total_cpu"
    DISK = "disk_io_each_moment"
    MEMORY = "total_memory_each_moment"
    ALL_PROCESSES = "processes_data"


# ======= Tables Column Names =======
class ProcessesColumns(str, Enum):
    TIME = "seconds_from_start"
    PROCESS_ID = "pid"
    PROCESS_NAME = "process_name"
    CPU_SUM_ACROSS_CORES = "cpu_percent_sum_across_cores"
    CPU_MEAN_ACROSS_CORES = "cpu_percent_mean_across_cores"
    NUMBER_OF_THREADS = "threads_num"
    USED_MEMORY = "used_memory_mb"
    MEMORY_PERCENT = "used_memory_percent"
    READ_COUNT = "disk_read_count"
    WRITE_COUNT = "disk_write_count"
    READ_BYTES = "disk_read_kb"
    WRITE_BYTES = "disk_write_kb"
    PAGE_FAULTS = "page_faults"
    BYTES_SENT = "network_kb_sent"
    PACKETS_SENT = "packets_sent"
    BYTES_RECEIVED = "network_kb_received"
    PACKETS_RECEIVED = "packets_received"
    PROCESS_OF_INTEREST = "process_of_interest"


class CPUColumns(str, Enum):
    TIME = "seconds_from_start"
    SUM_ACROSS_CORES_PERCENT = "sum_cpu_across_cores_percent"
    MEAN_ACROSS_CORES_PERCENT = "mean_cpu_across_cores_percent"
    # CORE = "Core"


class MemoryColumns(str, Enum):
    TIME = "seconds_from_start"
    USED_MEMORY = "total_memory_gb"
    USED_PERCENT = "total_memory_percent"


class DiskIOColumns(str, Enum):
    TIME = "seconds_from_start"
    READ_COUNT = "disk_read_count"
    WRITE_COUNT = "disk_write_count"
    READ_BYTES = "disk_read_kb"
    WRITE_BYTES = "disk_write_kb"
    READ_TIME = "disk_read_time"
    WRITE_TIME = "disk_write_time"


class NetworkIOColumns(str, Enum):
    TIME = "seconds_from_start"
    PACKETS_SENT = "packets_sent"
    PACKETS_RECEIVED = "packets_received"
    KB_SENT = "network_kb_sent"
    KB_RECEIVED = "network_kb_received"


class BatteryColumns(str, Enum):
    TIME = "seconds_from_start"
    PERCENTS = "battery_percent"
    CAPACITY = "battery_voltage_mV"
    VOLTAGE = "battery_remaining_capacity_mWh"


class LoggerName:
    SYSTEM_METRICS = "system_metrics"
    PROCESS_METRICS = "process_metrics"
    APPLICATION_FLOW = "system_metrics"


class IndexName:
    SYSTEM_METRICS = "system_metrics"
    PROCESS_METRICS = "process_metrics"
    APPLICATION_FLOW = "system_metrics"


def get_scanner_version_name(battery_monitor_type: BatteryMonitorType, process_monitor_type: ProcessMonitorType) -> str:
    if battery_monitor_type == battery_monitor_type.FULL and process_monitor_type == process_monitor_type.FULL:
        return "Full Process Monitoring Including Battery"
    elif battery_monitor_type == battery_monitor_type.FULL and process_monitor_type == process_monitor_type.PROCESSES_OF_INTEREST_ONLY:
        return "Only Process of Interest Monitoring Including Battery"
    elif battery_monitor_type == battery_monitor_type.WITHOUT_BATTERY and process_monitor_type == process_monitor_type.FULL:
        return "Full Process Monitoring Excluding Battery"
    elif battery_monitor_type == battery_monitor_type.WITHOUT_BATTERY and process_monitor_type == process_monitor_type.PROCESSES_OF_INTEREST_ONLY:
        return "Only Process of Interest Monitoring Excluding Battery"

    raise Exception("Scanner version is not supported")
