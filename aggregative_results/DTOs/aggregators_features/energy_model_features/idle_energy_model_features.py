from dataclasses import dataclass


@dataclass
class IdleEnergyModelFeatures:
    energy_per_second: float
    # cpu_usage_idle: float
    # memory_usage_idle: float
    # disk_read_bytes_usage_idle: float
    # disk_read_count_usage_idle: float
    # disk_write_bytes_usage_idle: float
    # disk_write_count_usage_idle: float
    # disk_read_time_idle_ms_sum: float
    # disk_write_time_idle_ms_sum: float
    # network_bytes_kb_sum_sent_idle: float
    # network_packets_sum_sent_idle: int
    # network_bytes_kb_sum_received_idle: float
    # network_packets_sum_received_idle: int
    #
    # total_energy_consumption_in_idle_mWh: float
    # number_of_page_faults_idle: float
    # duration_idle: float