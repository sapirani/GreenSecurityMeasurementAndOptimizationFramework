from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemEnergyModelFeatures:
    cpu_time_usage_system: float
    memory_gb_usage_system: float
    disk_read_kb_usage_system: float
    disk_read_count_usage_system: int
    disk_write_kb_usage_system: float
    disk_write_count_usage_system: int
    disk_read_time_system_ms_sum: float
    disk_write_time_system_ms_sum: float
    network_kb_sent_system: float
    network_packets_sent_system: int
    network_kb_received_system: float
    network_packets_received_system: int

    # necessary only for train-test and generating process consumption
    total_energy_consumption_system_mWh: Optional[float] = None
