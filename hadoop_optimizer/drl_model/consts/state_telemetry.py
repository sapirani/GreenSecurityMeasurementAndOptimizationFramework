from enum import Enum


class DRLTelemetryType(str, Enum):
    SYSTEM_CPU_INTEGRAL = "system_cpu_integral"
    SYSTEM_TOTAL_ENERGY_MWH = "system_total_energy_mwh"
    SYSTEM_CPU_ENERGY_MWH = "system_cpu_energy_mwh"
    SYSTEM_RAM_ENERGY_MWH = "system_ram_energy_mwh"
    SYSTEM_DISK_READ_ENERGY_MWH = "system_disk_read_energy_mwh"
    SYSTEM_DISK_WRITE_ENERGY_MWH = "system_disk_write_energy_mwh"
    SYSTEM_NETWORK_RECEIVED_ENERGY_MWH = "system_network_received_energy_mwh"
    SYSTEM_NETWORK_SENT_ENERGY_MWH = "system_network_sent_energy_mwh"