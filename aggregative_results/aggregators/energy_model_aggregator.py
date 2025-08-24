from dataclasses import asdict
from typing import Union, Optional

import pandas as pd

from aggregative_results.DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from aggregative_results.DTOs.aggregators_features.energy_model_features import EnergyModelFeatures
from aggregative_results.DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata
from measurements_model.energy_model import EnergyModel


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__previous_sample: Optional[EnergyModelFeatures] = None
        self.__model = EnergyModel()

    def extract_features(self, raw_results: ProcessSystemRawResults,
                         iteration_metadata: IterationMetadata) -> EnergyModelFeatures:
        process_data = raw_results.process_raw_results
        system_data = raw_results.system_raw_results
        return EnergyModelFeatures(
            timestamp=iteration_metadata.timestamp,
            desired_process_cpu=process_data.cpu_percent_sum_across_cores,
            total_system_cpu=system_data.cpu_percent_sum_across_cores,
            desired_process_memory_mb=process_data.used_memory_mb,
            total_system_memory_gb=system_data.total_memory_gb,
            desired_process_disk_read_count=process_data.disk_read_count,
            total_system_disk_read_count=system_data.disk_read_count,
            desired_process_disk_write_count=process_data.disk_write_count,
            total_system_disk_write_count=system_data.disk_write_count,
            desired_process_disk_read_kb=process_data.disk_read_kb,
            total_system_disk_read_kb=system_data.disk_read_kb,
            desired_process_disk_write_kb=process_data.disk_write_kb,
            total_system_disk_write_kb=system_data.disk_write_kb,
            desired_process_network_kb_sent=process_data.network_kb_sent,
            total_system_network_kb_sent=system_data.network_kb_sent,
            desired_process_packets_sent=process_data.packets_sent,
            total_system_packets_sent=system_data.packets_sent,
            desired_process_network_kb_received=process_data.network_kb_received,
            total_system_packets_received=system_data.packets_received,
            desired_process_packets_received=process_data.packets_received,
            total_system_network_kb_received=system_data.network_kb_received
        )

    def process_sample(self, sample: EnergyModelFeatures) -> Union[EnergyModelResult, EmptyAggregationResults]:
        try:
            if self.__previous_sample is None:
                return EmptyAggregationResults()

            sample_as_df = pd.DataFrame([asdict(sample)])
            sample_as_df["Duration"] = sample.timestamp - sample.timestamp
            energy_prediction = self.__model.predict(sample_as_df)
            return EnergyModelResult(energy_mwh=energy_prediction)
        except Exception as e:
            return EmptyAggregationResults()
        finally:
            self.__previous_sample = sample
