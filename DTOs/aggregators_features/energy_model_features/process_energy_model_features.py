from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessEnergyModelFeatures:
    cpu_time_usage_process: float
    memory_mb_usage_process: float
    disk_read_kb_usage_process: float
    disk_read_count_usage_process: int
    disk_write_kb_usage_process: float
    disk_write_count_usage_process: int
    number_of_page_faults_process: int
    network_kb_sent_process: float
    network_packets_sent_process: int
    network_kb_received_process: float
    network_packets_received_process: int

    # only for gt
    energy_consumption_process_mWh: Optional[float] = None
