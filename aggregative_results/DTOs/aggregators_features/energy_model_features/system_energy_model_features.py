from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemEnergyModelFeatures:
    cpu_usage_system: float
    memory_gb_usage_system: float
    disk_read_bytes_kb_usage_system: float
    disk_read_count_usage_system: int
    disk_write_bytes_kb_usage_system: float
    disk_write_count_usage_system: int
    disk_read_time_system_ms_sum: float
    disk_write_time_system_ms_sum: float
    number_of_page_faults_system: int
    network_bytes_kb_sum_sent_system: float
    network_packets_sum_sent_system: int
    network_bytes_kb_sum_received_system: float
    network_packets_sum_received_system: int

    # necessary only for train-test and generating process consumption
    total_energy_consumption_system_mWh: Optional[float] = None
