from enum import Enum
from hadoop_optimizer.drl_telemetry.consts.general import SYSTEM_PREFIX


class DRLTelemetryType(str, Enum):
    SYSTEM_CPU_INTEGRAL = f"{SYSTEM_PREFIX}cpu_integral"
    SYSTEM_TOTAL_ENERGY_MWH = f"{SYSTEM_PREFIX}system_total_energy_mwh"
    SYSTEM_CPU_ENERGY_MWH = f"{SYSTEM_PREFIX}cpu_energy_mwh"
    SYSTEM_RAM_ENERGY_MWH = f"{SYSTEM_PREFIX}ram_energy_mwh"
    SYSTEM_DISK_READ_ENERGY_MWH = f"{SYSTEM_PREFIX}disk_read_energy_mwh"
    SYSTEM_DISK_WRITE_ENERGY_MWH = f"{SYSTEM_PREFIX}write_energy_mwh"
    SYSTEM_NETWORK_RECEIVED_ENERGY_MWH = f"{SYSTEM_PREFIX}network_received_energy_mwh"
    SYSTEM_NETWORK_SENT_ENERGY_MWH = f"{SYSTEM_PREFIX}network_sent_energy_mwh"
    SYSTEM_MEMORY_PERCENT = f"{SYSTEM_PREFIX}total_memory_percent"
    SYSTEM_DISK_READ_COUNT = f"{SYSTEM_PREFIX}disk_read_count"
    SYSTEM_DISK_WRITE_COUNT = f"{SYSTEM_PREFIX}disk_write_count"
    SYSTEM_DISK_READ_KB = f"{SYSTEM_PREFIX}disk_read_kb"
    SYSTEM_DISK_WRITE_KB = f"{SYSTEM_PREFIX}disk_write_kb"
    SYSTEM_DISK_READ_TIME = f"{SYSTEM_PREFIX}disk_read_time"
    SYSTEM_DISK_WRITE_TIME = f"{SYSTEM_PREFIX}disk_write_time"
    SYSTEM_PACKETS_SENT = f"{SYSTEM_PREFIX}packets_sent"
    SYSTEM_PACKETS_RECEIVED = f"{SYSTEM_PREFIX}packets_received"
    SYSTEM_NETWORK_SENT_KB = f"{SYSTEM_PREFIX}network_kb_sent"
    SYSTEM_NETWORK_RECEIVED_KB = f"{SYSTEM_PREFIX}network_kb_received"
