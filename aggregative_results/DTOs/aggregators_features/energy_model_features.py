from dataclasses import dataclass
from datetime import datetime


@dataclass
class EnergyModelFeatures:
    timestamp: datetime

    desired_process_cpu: float
    total_system_cpu: float

    desired_process_memory_mb: float
    total_system_memory_gb: float

    desired_process_disk_read_count: int
    total_system_disk_read_count: int

    desired_process_disk_write_count: int
    total_system_disk_write_count: int

    desired_process_disk_read_kb: float
    total_system_disk_read_kb: float

    desired_process_disk_write_kb: float
    total_system_disk_write_kb: float

    desired_process_network_kb_sent: float
    total_system_network_kb_sent: float

    desired_process_packets_sent: int
    total_system_packets_sent: int

    desired_process_network_kb_received: float
    total_system_network_kb_received: float

    desired_process_packets_received: int
    total_system_packets_received: int
