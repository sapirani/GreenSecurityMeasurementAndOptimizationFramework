import logging
from typing import Union

import pandas as pd

from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregators.abstract_aggregator import T
from aggregators.energy_model_aggregators.abstract_energy_model_aggregator import EnergyModelAggregator
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator
from energy_model.models.aggregations_energy_model import AggregationsEnergyModel, ModelType
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from energy_model.energy_model_utils.resource_energy_calculator import ResourceEnergyCalculator
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy

logger = logging.getLogger(__name__)

class ProcessEnergyModelAggregator(EnergyModelAggregator):
    def __init__(self):
        super().__init__(ModelType.ProcessBased)

    def extract_features(self, raw_results: ProcessSystemRawResults,
                         iteration_metadata: IterationMetadata) -> Union[EnergyModelFeatures, EmptyFeatures]:

        return self.__energy_model_feature_extractor.extract_energy_model_features(raw_results,
                                                                                   iteration_metadata.timestamp)

    def process_sample(self, sample: Union[EnergyModelFeatures, EmptyFeatures]) -> Union[
        EnergyModelResult, EmptyAggregationResults]:
        return self._process_sample(sample)

    def _calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        return self.__resource_energy_calculator.calculate_relative_energy_consumption(sample.process_features,
                                                                                       energy_prediction)
    def _create_dataset_from_features(self, sample: T) -> pd.DataFrame:
        return EnergyModelConvertor.convert_features_to_pandas(sample)