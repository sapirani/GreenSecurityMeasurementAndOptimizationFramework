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

MINUTE = 60

pc_types = ["Unspecified", "Desktop", "Mobile Device", "Workstation",
                "EnterpriseServer", "SOHOServer", "AppliancePC", "PerformanceServer"]

# TODO: check if the list is correct
#physical_memory_types = ["Unknown", "Other", "DRAM", "Synchronous DRAM", "Cache DRAM", "EDO", "EDRAM", "VRAM",
#                         "SRAM", "RAM", "ROM", "Flash", "EEPROM", "FEPROM", "EPROM", "CDRAM", "3DRAM", "SDRAM",
#                         "SGRAM", "RDRAM", "DDR", "DDR2", "DDR2 FB-DIMM", "Unknown", "DDR3", "FBD2", "DDR4"]

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
    NO_SCAN = 1
    ANTIVIRUS = 2
    DummyANTIVIRUS = 3
    IDS = 4
    Perfmon = 5
    UserActivity = 6


class ScanMode(Enum):
    ONE_SCAN = 1
    CONTINUOUS_SCAN = 2
    

class PowerPlan:
    BALANCED = ("Balanced Plan", "381b4222-f694-41f0-9685-ff5bb260df2e")
    POWER_SAVER = ("Power Saver Plan", "a1841308-3541-4fab-bc81-f71556f20b4a")
    HIGH_PERFORMANCE = ("High Performance Plan", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")


class ScanType:
    FULL_SCAN = "FullScan"
    QUICK_SCAN = "QuickScan"
    CUSTOM_SCAN = "CustomScan"
    
    
class IdsType:
    SURICATA = "Suricata"
    SNORT = "Snort"


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
    READ_TIME = "READ(ms)"
    WRITE_TIME = "WRITE(ms)"


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


def get_core_name(core_number):
    return f"{CPUColumns.CORE} {core_number}(%)"
