from dataclasses import dataclass
from typing import Optional

# todo: remove when moving permanently to the new approach.
@dataclass
class SummaryColumns:
    total_column: str
    cpu_column: str
    memory_column: str
    disk_read_bytes_column: str
    disk_read_count_column: str
    disk_write_bytes_column: str
    disk_write_count_column: str
    disk_read_time_column: str
    disk_write_time_column: str
    number_of_page_faults_column: str
    duration_column: str
    total_energy_consumption_column: str

    network_bytes_kb_sum_sent_column: Optional[str] = None
    network_packets_sum_sent_column: Optional[str] = None
    network_bytes_kb_sum_received_column: Optional[str] = None
    network_packets_sum_received_column: Optional[str] = None
