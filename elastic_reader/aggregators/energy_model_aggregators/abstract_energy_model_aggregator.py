import logging
from abc import abstractmethod
from typing import Union
import pandas as pd
from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator, T
from energy_model.models.aggregations_energy_model import AggregationsEnergyModel, ModelType
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from energy_model.energy_model_utils.resource_energy_calculator import HardwareResourceEnergyCalculator
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy

logger = logging.getLogger(__name__)


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self, model_type: ModelType):
        self.__model = AggregationsEnergyModel.get_energy_model_instance(model_type)
        if self.__model is None:
            raise RuntimeError(f"Energy Model for type {model_type.name} not found.")
        self._energy_model_feature_extractor = EnergyModelFeatureExtractor()
        self._hardware_resource_energy_calculator = HardwareResourceEnergyCalculator()
        # todo: maybe support here the "DatasetProcessor" in order to change categorical columns, etc. Use when hardware columns are part of the train of the model

    @abstractmethod
    def extract_features(self, raw_results: Union[ProcessSystemRawResults, SystemRawResults], iteration_metadata: IterationMetadata) -> T:
        pass

    def _process_sample(self, sample: T) -> Union[EnergyModelResult, EmptyAggregationResults]:
        try:
            if isinstance(sample, EmptyFeatures):
                return EmptyAggregationResults()

            sample_df = self._convert_features_to_dataframe(sample)
            energy_prediction = self.__model.predict(sample_df)[0]
            if energy_prediction < 0:
                energy_prediction = 0
                logger.warning(
                    f"Received a negative value for energy prediction. Returning 0 mwh. The sample: {sample_df}")

            energy_per_resource = self._calculate_energy_per_resource(sample, energy_prediction)
            return EnergyModelResult(energy_mwh=energy_prediction,
                                     cpu_energy_consumption=energy_per_resource.cpu_energy_consumption,
                                     ram_energy_consumption=energy_per_resource.ram_energy_consumption,
                                     disk_io_read_energy_consumption=energy_per_resource.disk_io_read_energy_consumption,
                                     disk_io_write_energy_consumption=energy_per_resource.disk_io_write_energy_consumption,
                                     network_io_received_energy_consumption=energy_per_resource.network_io_received_energy_consumption,
                                     network_io_sent_energy_consumption=energy_per_resource.network_io_sent_energy_consumption)
        except Exception as e:
            logger.warning(
                "Error occurred when using the energy model. Returning empty aggregation results. \nThe error is: {}".format(
                    e))
            return EmptyAggregationResults()

    @abstractmethod
    def _calculate_energy_per_resource(self, sample: T, energy_prediction: float) -> SampleResourcesEnergy:
        pass

    @abstractmethod
    def _convert_features_to_dataframe(self, sample: T) -> pd.DataFrame:
        pass
