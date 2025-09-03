from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.relative_sample_features import \
    RelativeSampleFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from utils.general_consts import MB


class EnergyModelFeatureExtractor:
    @staticmethod
    def extract_process_features(process_data: ProcessRawResults,
                                 previous_sample_features: RelativeSampleFeatures,
                                 duration: float) -> ProcessEnergyModelFeatures:
        sum_cpu_usage_process = process_data.cpu_percent_sum_across_cores + previous_sample_features.cpu_usage_process
        cpu_integral_value = sum_cpu_usage_process * duration / 2
        process_cpu_time = cpu_integral_value / 100
        process_memory_relative_usage = process_data.used_memory_mb - previous_sample_features.memory_usage_process
        return ProcessEnergyModelFeatures(
            cpu_time_usage_process=process_cpu_time,
            memory_mb_usage_process=process_memory_relative_usage,
            disk_read_kb_usage_process=process_data.disk_read_kb,
            disk_write_kb_usage_process=process_data.disk_write_kb,
            disk_read_count_usage_process=process_data.disk_read_count,
            disk_write_count_usage_process=process_data.disk_write_count,
            number_of_page_faults_process=process_data.page_faults,
            network_kb_received_process=process_data.network_kb_received,
            network_packets_received_process=process_data.packets_received,
            network_kb_sent_process=process_data.network_kb_sent,
            network_packets_sent_process=process_data.packets_sent)

    @staticmethod
    def extract_system_features(system_data: SystemRawResults, previous_sample_features: RelativeSampleFeatures, duration: float) -> SystemEnergyModelFeatures:
        sum_cpu_usage_system = system_data.cpu_percent_sum_across_cores + previous_sample_features.cpu_usage_system
        cpu_integral_value = sum_cpu_usage_system * duration / 2
        system_cpu_time = cpu_integral_value / 100
        system_memory_relative_usage_gb = system_data.total_memory_gb - previous_sample_features.memory_usage_system
        system_memory_relative_usage_mb = system_memory_relative_usage_gb * MB
        return SystemEnergyModelFeatures(
            cpu_time_usage_system=system_cpu_time,
            memory_gb_usage_system=system_memory_relative_usage_mb,
            disk_read_kb_usage_system=system_data.disk_read_kb,
            disk_write_kb_usage_system=system_data.disk_write_kb,
            disk_read_count_usage_system=system_data.disk_read_count,
            disk_write_count_usage_system=system_data.disk_write_count,
            network_kb_sent_system=system_data.network_kb_sent,
            network_packets_sent_system=system_data.packets_sent,
            network_kb_received_system=system_data.packets_received,
            network_packets_received_system=system_data.packets_received,
            disk_read_time_system_ms_sum=system_data.disk_read_time,
            disk_write_time_system_ms_sum=system_data.disk_write_time
        )