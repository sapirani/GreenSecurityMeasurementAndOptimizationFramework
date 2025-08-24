class HardwareEnergyModelFeatures:
    PC_type: str
    PC_manufacturer: str
    system_family: str
    model: str
    machine_type: str
    device_name: str

    ## operating system info
    operating_system: str
    operating_system_release: str
    operating_system_version: str

    ## cpu info
    processor_name: str
    processor_physical_cores: int
    processor_total_cores: int
    processor_max_frequency: float
    processor_min_frequency: float

    ## RAM info
    total_ram: float
    physical_disk_name: str
    physical_disk_manufacturer: str
    physical_disk_model: str
    physical_disk_media_type: str
    disk_firmware_version: str
    logical_disk_name: str
    logical_disk_manufacturer: str
    logical_disk_model: str
    logical_disk_disk_type: str
    logical_disk_partition_style: str
    logical_disk_number_of_partitions: int
    physical_sector_size: int
    logical_sector_size: int
    bus_type: str
    file_system: str
    design_battery_capacity: float
    fully_charged_battery_capacity: float