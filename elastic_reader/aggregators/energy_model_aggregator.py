import logging
import threading
from typing import Union

from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator
from measurements_model.energy_model import EnergyModel
from measurements_model.energy_model_convertor import EnergyModelConvertor
from measurements_model.energy_model_feature_extractor import EnergyModelFeatureExtractor
from measurements_model.resource_energy_calculator import ResourceEnergyCalculator
from measurements_model.sample_resources_energy import SampleResourcesEnergy


class EnergyPerResourceConsts:
    """
    This class holds constant values that represent the energy consumption per one unit of a specific resource.
    For example, the energy usage for acquiring 1 MB of RAM is 17.18 mwh.
    """
    cpu_time_seconds = 1
    memory_gain_mb = 0.04 # todo: change
    memory_release_mb = 0.03  # todo: change to actual number
    disk_io_read_kbytes = 0.1261034238
    disk_io_write_kbytes = 0.1324211241
    network_received_kbytes = 0.1161303828
    network_sent_kbytes = 0.005866983801


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__model = EnergyModel.get_instance()
        self.__model.initialize_model()
        self.__resource_energy_calculator = ResourceEnergyCalculator(
            energy_per_cpu_time=EnergyPerResourceConsts.cpu_time_seconds,
            energy_per_gain_mb_ram=EnergyPerResourceConsts.memory_gain_mb,
            energy_per_release_mb_ram=EnergyPerResourceConsts.memory_release_mb,
            energy_per_disk_read_kb=EnergyPerResourceConsts.disk_io_read_kbytes,
            energy_per_disk_write_kb=EnergyPerResourceConsts.disk_io_write_kbytes,
            energy_per_network_received_kb=EnergyPerResourceConsts.network_received_kbytes,
            energy_per_network_sent_kb=EnergyPerResourceConsts.network_sent_kbytes
        )
        self.__energy_model_feature_extractor = EnergyModelFeatureExtractor()

    def extract_features(self, raw_results: ProcessSystemRawResults,
                         iteration_metadata: IterationMetadata) -> Union[EnergyModelFeatures, EmptyFeatures]:

        return self.__energy_model_feature_extractor.extract_energy_model_features(raw_results,
                                                                                   iteration_metadata.timestamp)

    def process_sample(self, sample: Union[EnergyModelFeatures, EmptyFeatures]) -> Union[
        EnergyModelResult, EmptyAggregationResults]:
        try:
            if isinstance(sample, EmptyFeatures):
                return EmptyAggregationResults()

            sample_df = EnergyModelConvertor.convert_features_to_pandas(sample)
            energy_prediction = self.__model.predict(sample_df)
            if energy_prediction < 0:
                energy_prediction = 0
                logging.warning(f"Received a negative value for energy prediction. Returning 0 mwh. The sample: {sample_df}")

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

    def __calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        return self.__resource_energy_calculator.calculate_relative_energy_consumption(sample.process_features, energy_prediction)
