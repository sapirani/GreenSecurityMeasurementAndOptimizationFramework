from enum import Enum
import psutil

# ======= Constants =======
GB = 2 ** 30
MB = 2 ** 20
KB = 2 ** 10
NEW_MEASUREMENT = -1
MEASUREMENT_NAME_DIR = "Measurement"
NUMBER_OF_CORES = psutil.cpu_count()

MINUTE = 60

pc_types = ["Unspecified", "Desktop", "Mobile Device", "Workstation",
                "EnterpriseServer", "SOHOServer", "AppliancePC", "PerformanceServer"]

# TODO: check if the list is correct
physical_memory_types = ["Unknown", "Other", "DRAM", "Synchronous DRAM", "Cache DRAM", "EDO", "EDRAM", "VRAM",
                         "SRAM", "RAM", "ROM", "Flash", "EEPROM", "FEPROM", "EPROM", "CDRAM", "3DRAM", "SDRAM",
                         "SGRAM", "RDRAM", "DDR", "DDR2", "DDR2 FB-DIMM", "Unknown", "DDR3", "FBD2", "DDR4"]


class ScanCommand:
    @staticmethod
    def antivirus(scan_type, custom_scan_path=None):
        custom_scan_query = "" if custom_scan_path == "" or custom_scan_path is None \
                               else f" -ScanPath {custom_scan_path}"
        return f"Start-MpScan -ScanType {scan_type}" + custom_scan_query


# ======= Static Classes =======
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


# ======= Tables Column Names =======
class BatteryColumns:
    TIME = "Time(sec)"
    PERCENTS = "REMAINING BATTERY(%)"
    CAPACITY = "REMAINING CAPACITY(mWh)"
    VOLTAGE = "Voltage(mV)"


class MemoryColumns:
    TIME = "Time(sec)"
    USED_MEMORY = "Used(GB)"
    USED_PERCENT = "Percentage"


class CPUColumns:
    TIME = "Time(sec)"
    USED_PERCENT = "Total CPU(%)"
    CORE = "Core"


class DiskIOColumns:
    TIME = "Time(sec)"
    READ_COUNT = "READ(#)"
    WRITE_COUNT = "WRITE(#)"
    READ_BYTES = "READ(KB)"
    WRITE_BYTES = "WRITE(KB)"


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
