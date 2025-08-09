from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional

from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator, EmptyAggregationResults


@dataclass
class CPUIntegralResult:
    cpu_integral: float


@dataclass
class CPUIntegralFeatures:
    date: datetime
    sum_cpu_usage_across_cores: float


class CPUIntegralAggregator(AbstractAggregator):
    def __init__(self):
        self.previous_sample: Optional[CPUIntegralFeatures] = None

    def extract_features(self, raw_input) -> CPUIntegralFeatures:
        pass

    def process_sample(self, sample: CPUIntegralFeatures) -> Union[CPUIntegralResult, EmptyAggregationResults]:
        if not self.previous_sample:
            self.previous_sample = sample
            return EmptyAggregationResults()

        # BUGFIX - UPDATE self.previous_sample

        delta_seconds = (sample.date - self.previous_sample.date).total_seconds()
        area = (sample.sum_cpu_usage_across_cores + self.previous_sample.sum_cpu_usage_across_cores) * delta_seconds / 2
        return CPUIntegralResult(cpu_integral=area)
