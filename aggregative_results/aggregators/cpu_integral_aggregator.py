from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional

from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs import ProcessRawResults
from aggregative_results.DTOs.aggregated_results_dtos import AggregationResult, EmptyAggregationResults
from aggregative_results.DTOs.raw_results_dtos import IterationMetadata, SystemRawResults


@dataclass
class CPUIntegralResult(AggregationResult):
    cpu_integral: float


@dataclass
class CPUIntegralFeatures:
    date: datetime
    cpu_percent_sum_across_cores: float


class CPUIntegralAggregator(AbstractAggregator):
    def __init__(self):
        self.previous_sample: Optional[CPUIntegralFeatures] = None

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
            if not self.previous_sample:
                return EmptyAggregationResults()

            delta_seconds = (sample.date - self.previous_sample.date).total_seconds()
            area = (sample.cpu_percent_sum_across_cores + self.previous_sample.cpu_percent_sum_across_cores) * delta_seconds / 2
            return CPUIntegralResult(cpu_integral=area)
        finally:
            self.previous_sample = sample
