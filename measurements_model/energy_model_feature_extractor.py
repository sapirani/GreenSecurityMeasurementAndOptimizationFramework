from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union

import pandas as pd

from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.relative_sample_features import RelativeSampleFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from utils.general_consts import MB


class EnergyModelFeatureExtractor:
    def __init__(self):
        self.__previous_sample: Optional[RelativeSampleFeatures] = None

    def extract_energy_model_features(self, raw_results: ProcessSystemRawResults,
                                      timestamp: datetime) -> Union[EnergyModelFeatures, EmptyFeatures]:
        if self.__previous_sample is None:
            return EmptyFeatures()

        duration = (timestamp - self.__previous_sample.timestamp).total_seconds()
        process_features = self.__extract_process_features(
            raw_results.process_raw_results,
            duration
        )

        system_features = self.__extract_system_features(
            raw_results.system_raw_results,
            duration
        )

        self.__set_previous_sample(raw_results, timestamp)
        return EnergyModelFeatures(
            duration=duration,
            process_features=process_features,
            system_features=system_features
        )

    def __set_previous_sample(self, raw_results: ProcessSystemRawResults, timestamp: datetime):
        self.__previous_sample = RelativeSampleFeatures(
            cpu_usage_process=raw_results.process_raw_results.cpu_percent_sum_across_cores,
            cpu_usage_system=raw_results.system_raw_results.cpu_percent_sum_across_cores,
            memory_usage_process=raw_results.process_raw_results.used_memory_mb,
            memory_usage_system=raw_results.system_raw_results.total_memory_gb,
            timestamp=timestamp
        )

    def __extract_process_features(self, process_data: ProcessRawResults,
                                   duration: float) -> ProcessEnergyModelFeatures:
        process_cpu_time = self.__calculate_integral_value(process_data.cpu_percent_sum_across_cores,
                                                           self.__previous_sample.cpu_usage_process,
                                                           duration) / 100
        process_memory_relative_usage = self.__calculate_relative_value(process_data.used_memory_mb,
                                                                        self.__previous_sample.memory_usage_process)
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

    def __extract_system_features(self, system_data: SystemRawResults, duration: float) -> SystemEnergyModelFeatures:
        system_cpu_time = self.__calculate_integral_value(system_data.cpu_percent_sum_across_cores,
                                                          self.__previous_sample.cpu_usage_system,
                                                          duration) / 100
        system_memory_relative_usage_mb = self.__calculate_relative_value(system_data.total_memory_gb,
                                                                          self.__previous_sample.memory_usage_system) * MB
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

    @classmethod
    def __calculate_integral_value(cls, current_val: float, previous_val: float, duration: float) -> float:
        return (current_val + previous_val) * duration / 2

    @classmethod
    def __calculate_relative_value(cls, current_val: float, previous_val: float) -> float:
        return current_val - previous_val

    def convert_features_to_pandas(self, sample: EnergyModelFeatures) -> pd.DataFrame:
        sample_as_dict = self.__convert_sample_to_dict(sample)
        sample_as_df = pd.DataFrame([sample_as_dict])
        return sample_as_df

    @classmethod
    def __convert_sample_to_dict(cls, sample: EnergyModelFeatures) -> dict[str, any]:
        sample_dict = {"duration": sample.duration,
                       **asdict(sample.process_features),
                       **asdict(sample.system_features)}
        if any(value is None for key, value in sample_dict.items() if key != "total_energy_consumption_system_mWh"):
            raise ValueError("Invalid sample, there is at least one empty field.")

        return sample_dict
