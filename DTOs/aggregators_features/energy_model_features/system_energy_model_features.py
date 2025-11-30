from dataclasses import dataclass


@dataclass
class SystemEnergyModelFeatures:
    duration: float
    cpu_seconds_system: float
    memory_mb_relative_system: float
    disk_read_kb_system: float
    disk_read_count_system: int
    disk_write_kb_system: float
    disk_write_count_system: int
    disk_read_time_ms_system: float
    disk_write_time_ms_system: float
    network_kb_sent_system: float
    network_packets_sent_system: int
    network_kb_received_system: float
    network_packets_received_system: int
