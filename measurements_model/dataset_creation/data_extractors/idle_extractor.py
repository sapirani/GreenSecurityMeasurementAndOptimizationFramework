import os

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor

SUMMARY_CSV = fr"summary.xlsx"

class IdleExtractor:
    def __init__(self, summary_extractor: AbstractSummaryExtractor):
        self.__summary_extractor = summary_extractor

    def extract(self, idle_summary_dir_path: str) -> IdleEnergyModelFeatures:
        idle_summary_file_path = os.path.join(idle_summary_dir_path, SUMMARY_CSV)
        idle_summary_results = self.__summary_extractor.extract_system_data(idle_summary_file_path)
        return IdleEnergyModelFeatures(
            cpu_usage_idle=idle_summary_results.cpu_usage_system,
            memory_usage_idle=idle_summary_results.cpu_usage_system,
            disk_read_bytes_usage_idle=idle_summary_results.disk_read_bytes_kb_usage_system,
            disk_read_count_usage_idle=idle_summary_results.disk_read_count_usage_system,
            disk_write_bytes_usage_idle=idle_summary_results.disk_write_bytes_kb_usage_system,
            disk_write_count_usage_idle=idle_summary_results.disk_write_count_usage_system,
            disk_read_time_idle_ms_sum=idle_summary_results.disk_read_time_system_ms_sum,
            disk_write_time_idle_ms_sum=idle_summary_results.disk_write_time_system_ms_sum,
            network_bytes_kb_sum_sent_idle=idle_summary_results.network_bytes_kb_sum_sent_system,
            network_packets_sum_sent_idle=idle_summary_results.network_packets_sum_sent_system,
            network_bytes_kb_sum_received_idle=idle_summary_results.network_bytes_kb_sum_received_system,
            network_packets_sum_received_idle=idle_summary_results.network_packets_sum_received_system,
            total_energy_consumption_in_idle_mWh=idle_summary_results.total_energy_consumption_system_mWh,
            number_of_page_faults_idle=idle_summary_results.number_of_page_faults_system,
            duration_idle=idle_summary_results.duration_system)
