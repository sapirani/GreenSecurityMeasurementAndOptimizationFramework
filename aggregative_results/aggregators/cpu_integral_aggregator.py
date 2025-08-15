from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional

from aggregative_results.DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata


@dataclass
class CPUIntegralResult(AbstractAggregationResult):
    cpu_integral: float


@dataclass
class CPUIntegralFeatures:
    date: datetime
    cpu_percent_sum_across_cores: float


class CPUIntegralAggregator(AbstractAggregator):
    def __init__(self):
        self._previous_sample: Optional[CPUIntegralFeatures] = None

    def extract_features(
            self,
            raw_results: SystemRawResults | ProcessRawResults,
            iteration_metadata: IterationMetadata
    ) -> CPUIntegralFeatures:
        return CPUIntegralFeatures(
            date=iteration_metadata.timestamp,
            cpu_percent_sum_across_cores=raw_results.cpu_percent_sum_across_cores
        )

    def process_sample(self, sample: CPUIntegralFeatures) -> Union[CPUIntegralResult, EmptyAggregationResults]:
        try:
            if not self._previous_sample:
                return EmptyAggregationResults()

            delta_seconds = (sample.date - self._previous_sample.date).total_seconds()
            area = (sample.cpu_percent_sum_across_cores + self._previous_sample.cpu_percent_sum_across_cores) * delta_seconds / 2
            return CPUIntegralResult(cpu_integral=area)
        finally:
            self._previous_sample = sample
