class ProcessColumns:
    CPU_PROCESS_COL = "cpu_usage_seconds_process"
    MEMORY_PROCESS_COL = "memory_mb_relative_process"
    DISK_READ_BYTES_PROCESS_COL = "disk_read_kb_process"
    DISK_READ_COUNT_PROCESS_COL = "disk_read_count_process"
    DISK_WRITE_BYTES_PROCESS_COL = "disk_write_kb_process"
    DISK_WRITE_COUNT_PROCESS_COL = "disk_write_count_process"
    PAGE_FAULT_PROCESS_COL = "number_of_page_faults_process"
    NETWORK_BYTES_SENT_PROCESS_COL = "network_kb_sent_process"
    NETWORK_PACKETS_SENT_PROCESS_COL = "network_packets_sent_process"
    NETWORK_BYTES_RECEIVED_PROCESS_COL = "network_kb_received_process"
    NETWORK_PACKETS_RECEIVED_PROCESS_COL = "network_packets_received_process"
    PROCESS_ID_COL = "pid"
    ENERGY_USAGE_PROCESS_COL = "energy_consumption_process_mWh"


class SystemColumns:
    CPU_SYSTEM_COL = "cpu_seconds_system"
    MEMORY_SYSTEM_COL = "memory_mb_relative_system"
    DISK_READ_BYTES_SYSTEM_COL = "disk_read_kb_system"
    DISK_READ_COUNT_SYSTEM_COL = "disk_read_count_system"
    DISK_WRITE_BYTES_SYSTEM_COL = "disk_write_kb_system"
    DISK_WRITE_COUNT_SYSTEM_COL = "disk_write_count_system"
    DISK_READ_TIME = "disk_read_time_ms_system"
    DISK_WRITE_TIME = "disk_write_time_ms_system"
    NETWORK_BYTES_SENT_SYSTEM_COL = "network_kb_sent_system"
    NETWORK_PACKETS_SENT_SYSTEM_COL = "network_packets_sent_system"
    NETWORK_BYTES_RECEIVED_SYSTEM_COL = "network_kb_received_system"
    NETWORK_PACKETS_RECEIVED_SYSTEM_COL = "network_packets_received_system"
    BATTERY_CAPACITY_MWH_SYSTEM_COL = "battery_capacity_mwh_system"

    DURATION_COL = "duration"
    ENERGY_USAGE_PER_SECOND_SYSTEM_COL = "energy_consumption_per_second_system_mWh"
    BATCH_ID_COL = "batch_id"
    SESSION_ID_COL = "session_id"
    HOSTNAME_COL = "hostname"
    ENERGY_RATIO_SHARE = "energy_share"
    ENERGY_USAGE_SYSTEM_COL = "energy_consumption_system_mWh"


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
