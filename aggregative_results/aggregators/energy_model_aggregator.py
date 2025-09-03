import threading
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
    memory = 17.18771578
    disk_io_read_kbytes = 0.1261034238
    disk_io_write_kbytes = 0.1324211241
    network_received_kbytes = 0.1161303828
    network_sent_kbytes = 0.005866983801


class EnergyModelAggregator(AbstractAggregator):
    __instance = None
    __lock = threading.Lock()
    __model = None
    __resource_energy_calculator = None
    __previous_sample: Optional[EnergyModelFeatures] = None

    def __init__(self):
        raise RuntimeError("This is a Singleton. Invoke get_instance() instead.")

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)
                    cls.__model = EnergyModel.get_instance()
                    cls.__resource_energy_calculator = ResourceEnergyCalculator(
                        energy_per_cpu=EnergyPerResourceConsts.cpu,
                        energy_per_mb_ram=EnergyPerResourceConsts.memory,
                        energy_per_disk_read_kb=EnergyPerResourceConsts.disk_io_read_kbytes,
                        energy_per_disk_write_kb=EnergyPerResourceConsts.disk_io_write_kbytes,
                        energy_per_network_received_kb=EnergyPerResourceConsts.network_received_kbytes,
                        energy_per_network_write_kb=EnergyPerResourceConsts.network_sent_kbytes
                    )

        return cls.__instance

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
            number_of_page_faults_process=process_data.page_faults,
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

            duration = sample.timestamp - self.__previous_sample.timestamp

            current_cpu_usage_system = sample.system_features.cpu_usage_system - self.__previous_sample.system_features.cpu_usage_system
            current_cpu_usage_process = sample.process_features.cpu_usage_process - self.__previous_sample.process_features.cpu_usage_process

            current_memory_usage_system = sample.system_features.memory_gb_usage_system - self.__previous_sample.system_features.memory_gb_usage_system
            current_memory_usage_process = sample.process_features.memory_mb_usage_process - self.__previous_sample.process_features.memory_mb_usage_process

            sample.process_features.cpu_usage_process = current_cpu_usage_process
            sample.process_features.memory_mb_usage_process = current_memory_usage_process
            sample.system_features.cpu_usage_system = current_cpu_usage_system
            sample.system_features.memory_gb_usage_system = current_memory_usage_system

            sample_as_dict = self.__convert_sample_to_dict(sample)
            sample_as_dict[DURATION_COLUMN] = duration
            sample_as_df = pd.DataFrame([sample_as_dict])

            # todo: add conversion of system memory GB into MB before inserting to the model
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
        if any(value is None for value in sample_dict.values()):
            raise ValueError("Invalid sample, there is at least one empty field.")

        return sample_dict

    def __calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        cpu_energy = self.__resource_energy_calculator.calculate_cpu_energy(sample.process_features.cpu_usage_process)
        memory_energy = self.__resource_energy_calculator.calculate_mb_ram_energy(
            sample.process_features.memory_mb_usage_process)
        disk_io_write_energy = self.__resource_energy_calculator.calculate_disk_write_kb_energy(
            sample.process_features.disk_write_bytes_kb_usage_process)
        disk_io_read_energy = self.__resource_energy_calculator.calculate_disk_read_kb_energy(
            sample.process_features.disk_read_bytes_kb_usage_process)
        network_received_energy = self.__resource_energy_calculator.calculate_network_received_kb_energy(
            sample.process_features.network_bytes_sum_kb_received_process)
        network_sent_energy = self.__resource_energy_calculator.calculate_network_sent_kb_energy(
            sample.process_features.network_packets_sum_sent_process)

        per_resource_energy_sum = cpu_energy + memory_energy + disk_io_write_energy + disk_io_read_energy + network_received_energy + network_sent_energy
        return SampleResourcesEnergy(
            cpu_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(cpu_energy,
                                                                                             per_resource_energy_sum,
                                                                                             energy_prediction),
            ram_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(memory_energy,
                                                                                             per_resource_energy_sum,
                                                                                             energy_prediction),
            disk_io_read_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(
                disk_io_read_energy, per_resource_energy_sum, energy_prediction),
            disk_io_write_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(
                disk_io_write_energy, per_resource_energy_sum, energy_prediction),
            network_io_received_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(
                network_received_energy, per_resource_energy_sum, energy_prediction),
            network_io_sent_energy_consumption=self.__resource_energy_calculator.per_resource_energy_sum(
                network_sent_energy, per_resource_energy_sum, energy_prediction)
        )
