from enum import Enum
import psutil

# ======= Constants =======
ANTIVIRUS_PROCESS_NAME = "MsMpEng.exe"
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
class ProgramInterface:
    def get_program_name(self) -> str:
        pass

    def get_process_name(self) -> str:
        pass

    def get_command(self) -> str:
        pass

    def path_adjustments(self) -> str:
        return ""


class AntivirusProgram(ProgramInterface):
    def __init__(self, scan_type, custom_scan_path):
        if custom_scan_path == "" or custom_scan_path == '""':
            self.custom_scan_path = None
        else:
            self.custom_scan_path = custom_scan_path
        self.scan_type = scan_type

    def get_program_name(self):
        return "Windows Defender"

    def get_command(self) -> str:
        custom_scan_query = "" if self.custom_scan_path is None else f" -ScanPath {self.custom_scan_path}"
        return f"Start-MpScan -ScanType {self.scan_type}" + custom_scan_query

    def path_adjustments(self):
        return self.scan_type


class DummyAntivirusProgram(ProgramInterface):
    def __init__(self, scan_path):
        self.scan_path = scan_path

    def get_program_name(self):
        return "Dummy Antivirus"

    def get_command(self) -> str:
        return f"python FilesReader.py {self.scan_path}"


class IDSProgram(ProgramInterface):
    def __init__(self, ids_type, interface_name, log_dir, installation_dir="C:\Program Files"):
        self.ids_type = ids_type
        self.interface_name = interface_name
        self.log_dir = log_dir
        self.installation_dir = installation_dir

    def get_program_name(self):
        return "IDS"

    def get_command(self):
        return rf"& '{self.installation_dir}\{self.ids_type}\{self.ids_type.lower()}.exe' -i {self.interface_name} -l '{self.installation_dir}\{self.ids_type}\{self.log_dir}'"


class ProgramToScan(Enum):
    ANTIVIRUS = 0
    DummyANTIVIRUS = 1
    IDS = 2


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
