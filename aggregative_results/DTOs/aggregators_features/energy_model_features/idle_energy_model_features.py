class IdleEnergyModelFeatures:
    cpu_usage_idle: float
    memory_usage_idle: float
    disk_read_bytes_usage_idle: float
    disk_read_count_usage_idle: float
    disk_write_bytes_usage_idle: float
    disk_write_count_usage_idle: float
    disk_read_time_idle_ms_sum: float
    disk_write_time_idle_ms_sum: float

    total_energy_consumption_in_idle_mWh: float
    number_of_page_faults_idle: float
    duration_idle: float