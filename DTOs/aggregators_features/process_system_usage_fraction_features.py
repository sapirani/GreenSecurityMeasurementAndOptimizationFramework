from dataclasses import dataclass
from typing import List


@dataclass
class ProcessSystemUsageFractionFeatures:
    desired_process_cpu: float
    processes_cpu: List[float]

    desired_process_memory_mb: float
    processes_memory_mb: List[float]

    desired_process_disk_read_count: int
    processes_disk_read_count: List[int]

    desired_process_disk_write_count: int
    processes_disk_write_count: List[int]

    desired_process_disk_read_kb: float
    processes_disk_read_kb: List[float]

    desired_process_disk_write_kb: float
    processes_disk_write_kb: List[float]

    desired_process_network_kb_sent: float
    processes_network_kb_sent: List[float]

    desired_process_packets_sent: int
    processes_packets_sent: List[int]

    desired_process_network_kb_received: float
    processes_network_kb_received: List[float]

    desired_process_packets_received: int
    processes_packets_received: List[int]
