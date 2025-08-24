SCORING_METHODS_FOR_MODEL = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

NO_ENERGY_MEASURED = -1

# Paths
IDLE_DIR_PATH = r"C:\Users\Administrator\Desktop\green security\tmp - idle\Measurement 1"

ALL_MEASUREMENTS_SYSTEM_RESOURCE_ISOLATION_VERSION_NO_NETWORK_DIR_PATH = r"C:\Users\Administrator\Desktop\green security\tmp"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\measurements_with_resources_dudu_summary"
ALL_MEASUREMENTS_NATIVE_VERSION_WITH_NETWORK_DIR_PATH = r"C:\Users\Administrator\Desktop\green security\tmp"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\measurements_other_summary"

FULL_DATASET_PATH = r"C:\Users\Administrator\Desktop\green security\results\full_dataset.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_dataset.csv"
FULL_PREPROCESSED_DATASET_PATH = r"C:\Users\Administrator\Desktop\green security\results\full_processed_dataset.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_preprocessed_dataset.csv"
DATASET_AFTER_FEATURE_SELECTION_PATH = r"C:\Users\Administrator\Desktop\green security\results\dataset_after_feature_selection.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\dataset_after_feature_selection.csv"

DF_ALL_FEATURES_NO_ENERGY_PATH = r"C:\Users\Administrator\Desktop\green security\results\dataset_no_energy.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\all_features_no_energy_preprocessed_dataset.csv"
DF_WITHOUT_SYSTEM_PATH = r"C:\Users\Administrator\Desktop\green security\results\dataset_no_system.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\without_system_preprocessed_dataset.csv"
DF_PROCESS_AND_FULL_SYSTEM_PATH = r"C:\Users\Administrator\Desktop\green security\results\dataset_process_and_system.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\process_and_full_system_preprocessed_dataset.csv"
DF_WITHOUT_HARDWARE_PATH = r"C:\Users\Administrator\Desktop\green security\results\dataset_no_hardware.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\without_hardware_preprocessed_dataset.csv"

TRAIN_SET_PATH = r"C:\Users\Administrator\Desktop\green security\results\trainset.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\train_set.csv"
TEST_SET_PATH = r"C:\Users\Administrator\Desktop\green security\results\testset.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\test_set.csv"

GRID_SEARCH_TEST_RESULTS_PATH = r"C:\Users\Administrator\Desktop\green security\results\grid_search_results.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\best_estimators_results.csv"
RESULTS_TOP_MODELS_PATH = r"C:\Users\Administrator\Desktop\green security\results\top_estimators_results.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\top_estimators_results"


class SummaryFieldsNativeVersion:
    DURATION = "Duration"
    CPU = "CPU"
    MEMORY = "Memory (MB)"
    PAGE_FAULTS = "Page Faults"
    DISK_IO_READ_BYTES = "Disk IO Read (KB - sum)"
    DISK_IO_READ_COUNT = "Disk IO Read Count (# - sum)"
    DISK_IO_WRITE_BYTES = "Disk IO Write (KB - sum)"
    DISK_IO_WRITE_COUNT = "Disk IO Write Count (# - sum)"
    DISK_IO_READ_TIME = "Disk IO Read Time (ms - sum)"
    DISK_IO_WRITE_TIME = "Disk IO Write Time (ms - sum)"
    NETWORK_SENT_TOTAL = "Network Size Sent (KB - sum)"
    NETWORK_SENT_PACKET_COUNT = "Network Packets Sent (# - sum)"
    NETWORK_RECEIVED_TOTAL = "Network Size Received (KB - sum)"
    NETWORK_RECEIVED_PACKET_COUNT = "Network Packets Received (# - sum)"
    ENERGY_CONSUMPTION = "Energy consumption - total energy(mwh)"
    TOTAL_COLUMN = "Total"
    PROCESS_COLUMN = "ClamAV"  # TODO: change according to the process


class SummaryFieldsSystemResourcesIsolationVersion:
    DURATION = "Duration"
    CPU_PROCESS = "CPU Process"
    CPU_SYSTEM = "CPU System (total - process)"
    MEMORY_PROCESS = "Memory Process (MB)"
    MEMORY_SYSTEM = "Memory Total (total - process) (MB)"
    PAGE_FAULTS = "Page Faults"
    IO_READ_BYTES_PROCESS = "IO Read Process (KB - sum)"
    IO_READ_BYTES_SYSTEM = "IO Read System (total - process) (KB - sum)"
    IO_READ_COUNT_PROCESS = "IO Read Count Process (# - sum)"
    IO_READ_COUNT_SYSTEM = "IO Read Count System (total - process) (# - sum)"
    IO_WRITE_BYTES_PROCESS = "IO Write Process (KB - sum)"
    IO_WRITE_BYTES_SYSTEM = "IO Write System (total - process) (KB - sum)"
    IO_WRITE_COUNT_PROCESS = "IO Write Count Process (# - sum)"
    IO_WRITE_COUNT_SYSTEM = "IO Write Count System (total - process) (# - sum)"
    DISK_IO_READ_TIME = "Disk IO Read Time (ms - sum)"
    DISK_IO_WRITE_TIME = "Disk IO Write Time (ms - sum)"
    ENERGY_CONSUMPTION = "Energy consumption - total energy(mwh)"
    TOTAL_COLUMN = "System (total - all processes)"


# Processes File related
PROCESSES_FILE_NAME = "processes_data.csv"
HEAVYLOAD_PROCESS_NAME = "HeavyLoad.exe"


class AllProcessesFileFields:
    PROCESS_NAME_COL = "PNAME"
    CPU = "CPU(%)"
    MEMORY = "MEMORY(MB)"
    DISK_READ_BYTES = "READ_IO(KB)"
    DISK_READ_COUNT = "READ_IO(#)"
    DISK_WRITE_BYTES = "WRITE_IO(KB)"
    DISK_WRITE_COUNT = "WRITE_IO(#)"
    PAGE_FAULTS = "PAGE_FAULTS"
    NETWORK_BYTES_SENT = "NETWORK_DATA_SENT(KB)"
    NETWORK_PACKETS_SENT = "PACKETS_SENT(#)"
    NETWORK_BYTES_RECEIVED = "NETWORK_BYTES_RECEIVED(KB)"
    NETWORK_PACKETS_RECEIVED = "PACKETS_RECEIVED(#)"


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
    NETWORK_BYTES_SENT_PROCESS_COL = "network_bytes_sum_sent_process"
    NETWORK_PACKETS_SENT_PROCESS_COL = "network_packets_sum_sent_process"
    NETWORK_BYTES_RECEIVED_PROCESS_COL = "network_bytes_sum_received_process"
    NETWORK_PACKETS_RECEIVED_PROCESS_COL = "network_packets_sum_received_process"
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
    NETWORK_BYTES_SENT_SYSTEM_COL = "network_bytes_sum_sent_system"
    NETWORK_PACKETS_SENT_SYSTEM_COL = "network_packets_sum_sent_system"
    NETWORK_BYTES_RECEIVED_SYSTEM_COL = "network_bytes_sum_received_system"
    NETWORK_PACKETS_RECEIVED_SYSTEM_COL = "network_packets_sum_received_system"

    DURATION_COL = "duration_system"  # ??? maybe give up
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
    DURATION_COL = "duration_idle"  # ??


# *** Hardware Related Cols *** #

class HardwareColumns:
    ## machine info
    PC_TYPE = "PC_type"
    PC_MANUFACTURER = "PC_manufacturer"
    SYSTEM_FAMILY = "system_family"
    PC_MODEL = "model"
    MACHINE_TYPE = "machine_type"
    DEVICE_NAME = "device_name"

    ## operating system info
    OPERATING_SYSTEM = "operating_system"
    OPERATING_SYSTEM_RELEASE = "operating_system_release"
    OPERATING_SYSTEM_VERSION = "operating_system_version"

    ## cpu info
    PROCESSOR_NAME = "processor_name"
    PROCESSOR_PHYSICAL_CORES = "processor_physical_cores"
    PROCESSOR_TOTAL_CORES = "processor_total_cores"
    PROCESSOR_MAX_FREQ = "processor_max_frequency"
    PROCESSOR_MIN_FREQ = "processor_min_frequency"

    ## RAM info
    TOTAL_RAM = "total_ram"
    PHYSICAL_DISK_NAME = "physical_disk_name"
    PHYSICAL_DISK_MANUFACTURER = "physical_disk_manufacturer"
    PHYSICAL_DISK_MODEL = "physical_disk_model"
    PHYSICAL_DISK_MEDIA_TYPE = "physical_disk_media_type"
    PHYSICAL_DISK_FIRMWARE_VERSION = "disk_firmware_version"
    LOGICAL_DISK_NAME = "logical_disk_name"
    LOGICAL_DISK_MANUFACTURER = "logical_disk_manufacturer"
    LOGICAL_DISK_MODEL = "logical_disk_model"
    LOGICAL_DISK_DISK_TYPE = "logical_disk_disk_type"
    LOGICAL_DISK_PARTITION_STYLE = "logical_disk_partition_style"
    LOGICAL_DISK_NUMBER_OF_PARTITIONS = "logical_disk_number_of_partitions"
    PHYSICAL_SECTOR_SIZE = "physical_sector_size"
    LOGICAL_SECTOR_SIZE = "logical_sector_size"
    BUS_TYPE = "bus_type"
    FILESYSTEM = "file_system"
    BATTERY_DESIGN_CAPACITY = "design_battery_capacity"
    FULLY_CHARGED_BATTERY_CAPACITY = "fully_charged_battery_capacity"
