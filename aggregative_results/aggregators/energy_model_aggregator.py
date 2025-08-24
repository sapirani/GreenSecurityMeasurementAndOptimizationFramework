from dataclasses import asdict
from typing import Union, Optional

import pandas as pd

from aggregative_results.DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from aggregative_results.DTOs.aggregators_features.energy_model_features.full_energy_model_features import \
    EnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import \
    HardwareEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from aggregative_results.DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.system_resources_isolation_summary_extractor import \
    SystemResourcesIsolationSummaryExtractor
from measurements_model.energy_model import EnergyModel

DURATION_COLUMN = "duration"
DEFAULT_IDLE_SUMMARY_EXTRACTOR = SystemResourcesIsolationSummaryExtractor()
DEFAULT_IDLE_DIR = r"C:\Users\Administrator\Desktop\green security\tmp - idle\Measurement 1"


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__model = EnergyModel()
        self.__previous_sample: Optional[EnergyModelFeatures] = None
        self.__default_hardware_features = HardwareExtractor().extract("")
        self.__default_idle_features = IdleExtractor(DEFAULT_IDLE_SUMMARY_EXTRACTOR).extract(DEFAULT_IDLE_DIR)

    def extract_features(self, raw_results: ProcessSystemRawResults,
                         iteration_metadata: IterationMetadata) -> EnergyModelFeatures:
        process_data = raw_results.process_raw_results
        system_data = raw_results.system_raw_results

        process_features = ProcessEnergyModelFeatures(
            cpu_usage_process=process_data.cpu_percent_sum_across_cores,
            memory_mb_usage_process=process_data.used_memory_mb,
            disk_read_bytes_kb_usage_process=process_data.disk_read_kb,
            disk_write_bytes_kb_usage_process=process_data.disk_write_kb,
            disk_read_count_usage_process=process_data.disk_read_count,
            disk_write_count_usage_process=process_data.disk_write_count,
            network_bytes_sum_kb_received_process=process_data.network_kb_received,
            network_packets_sum_received_process=process_data.packets_received,
            network_bytes_sum_kb_sent_process=process_data.network_kb_sent,
            network_packets_sum_sent_process=process_data.packets_sent)
        system_features = SystemEnergyModelFeatures(
            cpu_usage_system=system_data.cpu_percent_sum_across_cores,
            memory_gb_usage_system=system_data.total_memory_gb,
            disk_read_bytes_kb_usage_system=system_data.disk_read_kb,
            disk_write_bytes_kb_usage_system=system_data.disk_write_kb,
            disk_read_count_usage_system=system_data.disk_read_count,
            disk_write_count_usage_system=system_data.disk_write_count,
            number_of_page_faults_system=5,  # todo: change
            network_bytes_kb_sum_sent_system=system_data.network_kb_sent,
            network_packets_sum_sent_system=system_data.packets_sent,
            network_bytes_kb_sum_received_system=system_data.packets_received,
            network_packets_sum_received_system=system_data.packets_received,
            disk_read_time_system_ms_sum=system_data.disk_read_time,
            disk_write_time_system_ms_sum=system_data.disk_write_time
        )
        return EnergyModelFeatures(
            timestamp=iteration_metadata.timestamp,
            process_features=process_features,
            system_features=system_features,
            hardware_features=self.__default_hardware_features,
            idle_features=self.__default_idle_features,
        )

    def process_sample(self, sample: EnergyModelFeatures) -> Union[EnergyModelResult, EmptyAggregationResults]:
        try:
            if self.__previous_sample is None:
                return EmptyAggregationResults()

            sample_as_dict = self.__convert_sample_to_dict(sample)
            duration = sample.timestamp - self.__previous_sample.timestamp
            sample_as_dict[DURATION_COLUMN] = duration

            sample_as_df = pd.DataFrame([sample_as_dict])
            energy_prediction = self.__model.predict(sample_as_df)
            return EnergyModelResult(energy_mwh=energy_prediction)
        except Exception as e:
            return EmptyAggregationResults()
        finally:
            self.__previous_sample = sample

    def __convert_sample_to_dict(self, sample: EnergyModelFeatures) -> dict[str, any]:
        sample_dict = {**asdict(sample.process_features), **asdict(sample.system_features)}
        if sample.idle_features is not None:
            sample_dict = {**sample_dict, **asdict(sample.idle_features)}
        if sample.hardware_features is not None:
            sample_dict = {**sample_dict, **asdict(sample.hardware_features)}
        return {key: value for key,value in sample_dict.items() if value is not None}