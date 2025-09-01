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
from measurements_model.resource_energy_calculator import ResourceEnergyCalculator
from measurements_model.sample_resources_energy import SampleResourcesEnergy

DURATION_COLUMN = "duration"
DEFAULT_IDLE_SUMMARY_EXTRACTOR = SystemResourcesIsolationSummaryExtractor()
DEFAULT_IDLE_DIR = r"C:\Users\Administrator\Desktop\green security\tmp - idle\Measurement 1"


class EnergyPerResourceConsts:
    # todo: add real consts
    cpu = 0.01
    memory = 0.02
    disk_io_read_bytes = 0.03
    disk_io_write_bytes = 0.04
    network_received_bytes = 0.05
    network_sent_bytes = 0.06


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__model = EnergyModel()
        self.__previous_sample: Optional[EnergyModelFeatures] = None
        self.__resource_energy_calculator = ResourceEnergyCalculator(
            energy_per_cpu=EnergyPerResourceConsts.cpu,
            energy_per_gb_ram=EnergyPerResourceConsts.memory,
            energy_per_disk_read_kb=EnergyPerResourceConsts.disk_io_read_bytes,
            energy_per_disk_write_kb=EnergyPerResourceConsts.disk_io_write_bytes,
            energy_per_network_received_kb=EnergyPerResourceConsts.network_received_bytes,
            energy_per_network_write_kb=EnergyPerResourceConsts.network_sent_bytes
        )

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
            system_features=system_features
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

            energy_per_resource = self.__calculate_energy_per_resource(sample, energy_prediction)
            return EnergyModelResult(energy_mwh=energy_prediction,
                                     cpu_energy_consumption=energy_per_resource.cpu_energy_consumption,
                                     ram_energy_consumption=energy_per_resource.ram_energy_consumption,
                                     disk_io_read_energy_consumption=energy_per_resource.disk_io_read_energy_consumption,
                                     disk_io_write_energy_consumption=energy_per_resource.disk_io_write_energy_consumption,
                                     network_io_received_energy_consumption=energy_per_resource.network_io_received_energy_consumption,
                                     network_io_sent_energy_consumption=energy_per_resource.network_io_sent_energy_consumption)
        except Exception as e:
            print(
                "Error occurred when using the energy model. Returning empty aggregation results. \nThe error is: {}".format(
                    e))
            return EmptyAggregationResults()
        finally:
            self.__previous_sample = sample

    def __convert_sample_to_dict(self, sample: EnergyModelFeatures) -> dict[str, any]:
        sample_dict = {**asdict(sample.process_features), **asdict(sample.system_features)}
        return {key: value for key, value in sample_dict.items() if value is not None}

    def __calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        cpu_energy = self.__resource_energy_calculator.calculate_cpu_energy(sample.process_features.cpu_usage_process)
        # todo: change memory to mb
        memory_energy = self.__resource_energy_calculator.calculate_gb_ram_energy(
            sample.process_features.memory_mb_usage_process)
        disk_io_write_energy = self.__resource_energy_calculator.calculate_disk_write_kb_energy(
            sample.process_features.disk_write_bytes_kb_usage_process)
        disk_io_read_energy = self.__resource_energy_calculator.calculate_disk_read_kb_energy(
            sample.process_features.disk_read_bytes_kb_usage_process)
        network_received_energy = self.__resource_energy_calculator.calculate_network_received_kb_energy(
            sample.process_features.network_bytes_sum_kb_received_process)
        network_sent_energy = self.__resource_energy_calculator.calculate_network_sent_kb_energy(
            sample.process_features.network_packets_sum_sent_process)

        total_resource_energy = cpu_energy + memory_energy + disk_io_write_energy + disk_io_read_energy + network_received_energy + network_sent_energy
        return SampleResourcesEnergy(
            cpu_energy_consumption=self.__resource_energy_calculator.get_energy_part(cpu_energy, total_resource_energy,
                                                                                     energy_prediction),
            ram_energy_consumption=self.__resource_energy_calculator.get_energy_part(memory_energy,
                                                                                     total_resource_energy,
                                                                                     energy_prediction),
            disk_io_read_energy_consumption=self.__resource_energy_calculator.get_energy_part(disk_io_read_energy,
                                                                                              total_resource_energy,
                                                                                              energy_prediction),
            disk_io_write_energy_consumption=self.__resource_energy_calculator.get_energy_part(disk_io_write_energy,
                                                                                               total_resource_energy,
                                                                                               energy_prediction),
            network_io_received_energy_consumption=self.__resource_energy_calculator.get_energy_part(
                network_received_energy, total_resource_energy, energy_prediction),
            network_io_sent_energy_consumption=self.__resource_energy_calculator.get_energy_part(network_sent_energy,
                                                                                                 total_resource_energy,
                                                                                                 energy_prediction)
        )
