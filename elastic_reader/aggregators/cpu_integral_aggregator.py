from typing import Union, Optional

from DTOs.aggregated_results_dtos.cpu_integral_result import CPUIntegralResult
from DTOs.aggregators_features.cpu_integral_features import CPUIntegralFeatures
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from elastic_reader.aggregators.aggregation_types import AggregationType
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator


class CPUIntegralAggregator(AbstractAggregator):
    def __init__(self):
        self.__previous_sample: Optional[CPUIntegralFeatures] = None

    @property
    def name(self) -> AggregationType:
        return AggregationType.CPUIntegral

    def extract_features(
            self,
            raw_results: Union[SystemRawResults, ProcessRawResults],
            iteration_metadata: IterationMetadata
    ) -> CPUIntegralFeatures:
        return CPUIntegralFeatures(
            date=iteration_metadata.timestamp,
            cpu_percent_sum_across_cores=raw_results.cpu_percent_sum_across_cores
        )

    def process_sample(self, sample: CPUIntegralFeatures) -> Union[CPUIntegralResult, EmptyAggregationResults]:
        try:
            if not self.__previous_sample:
                return EmptyAggregationResults()

            delta_seconds = (sample.date - self.__previous_sample.date).total_seconds()
            area = (sample.cpu_percent_sum_across_cores + self.__previous_sample.cpu_percent_sum_across_cores) * delta_seconds / 2
            return CPUIntegralResult(cpu_integral=area)
        finally:
            self.__previous_sample = sample
