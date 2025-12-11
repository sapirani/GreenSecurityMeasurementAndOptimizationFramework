import logging
from typing import Union
import pandas as pd
from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from elastic_reader.aggregators.aggregation_types import AggregationType
from elastic_reader.aggregators.energy_model_aggregators.abstract_energy_model_aggregator import EnergyModelAggregator
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy
from energy_model.models.aggregations_energy_model import ModelType

logger = logging.getLogger(__name__)


class SystemEnergyModelAggregator(EnergyModelAggregator):
    def __init__(self):
        super().__init__(ModelType.SystemBased)

    @property
    def name(self) -> AggregationType:
        return AggregationType.SystemEnergyModelAggregator

    def _calculate_energy_per_resource(self, sample: SystemEnergyModelFeatures,
                                       energy_prediction: float) -> SampleResourcesEnergy:
        return self._resource_energy_calculator.calculate_relative_energy_consumption_system(sample, energy_prediction)

    def _convert_features_to_dataframe(self, sample: SystemEnergyModelFeatures) -> pd.DataFrame:
        return EnergyModelConvertor.convert_system_features_to_pandas(sample)

    def extract_features(self, raw_results: SystemRawResults,
                         iteration_metadata: IterationMetadata) -> Union[SystemEnergyModelFeatures, EmptyFeatures]:
        return self._energy_model_feature_extractor.extract_system_energy_model_features(raw_results,
                                                                                         iteration_metadata.timestamp)

    def process_sample(self, sample: Union[SystemEnergyModelFeatures, EmptyFeatures]) -> Union[
        EnergyModelResult, EmptyAggregationResults]:
        return self._process_sample(sample)
