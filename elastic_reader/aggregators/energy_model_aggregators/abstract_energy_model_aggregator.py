import logging
from abc import abstractmethod
from typing import Union

import pandas as pd

from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator, T
from energy_model.models.aggregations_energy_model import AggregationsEnergyModel, ModelType
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from energy_model.energy_model_utils.resource_energy_calculator import ResourceEnergyCalculator
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy

logger = logging.getLogger(__name__)


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self, model_type: ModelType):
        self._model = AggregationsEnergyModel.get_instance()
        self._model.initialize_model(model_type)
        self.__resource_energy_calculator = ResourceEnergyCalculator()
        self.__energy_model_feature_extractor = EnergyModelFeatureExtractor()
        self.__model_type = model_type
        # todo: maybe support here the "DatasetProcessor" in order to change categorical columns, etc. Use when hardware columns are part of the train of the model

    @abstractmethod
    def _create_dataset_from_features(self, sample: T) -> pd.DataFrame:
        pass

    @abstractmethod
    def _calculate_energy_per_resource(self, sample: T,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        pass

    def _process_sample(self, sample: T) -> Union[
        EnergyModelResult, EmptyAggregationResults]:
        try:
            if isinstance(sample, EmptyFeatures):
                return EmptyAggregationResults()

            sample_df = self._create_dataset_from_features(sample)
            energy_prediction = self._model.predict(sample_df, self.__model_type)[0]
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

    def __calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        return self.__resource_energy_calculator.calculate_relative_energy_consumption(sample.process_features,
                                                                                       energy_prediction)
