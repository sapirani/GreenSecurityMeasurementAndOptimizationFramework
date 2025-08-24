from typing import Union, Optional

from aggregative_results.DTOs.aggregated_results_dtos.cpu_integral_result import CPUIntegralResult
from aggregative_results.DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from aggregative_results.DTOs.aggregators_features.energy_model_features import EnergyModelFeatures
from aggregative_results.DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata


class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__previous_sample: Optional[EnergyModelFeatures] = None
        self.__model = "my model" # todo: initialize model from file

    def extract_features(
            self,
            raw_results: ProcessSystemRawResults,
            iteration_metadata: IterationMetadata
    ) -> EnergyModelFeatures:
        return EnergyModelFeatures()

    def process_sample(self, sample: EnergyModelFeatures) -> Union[EnergyModelResult, EmptyAggregationResults]:
        try:
            if not self.__previous_sample:
                return EmptyAggregationResults()

            return EnergyModelResult(energy_mwh=5) # todo: add model calling
        finally:
            self.__previous_sample = sample
