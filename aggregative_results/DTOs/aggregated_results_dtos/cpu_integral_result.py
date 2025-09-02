from dataclasses import dataclass

from aggregative_results.DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult


@dataclass
class CPUIntegralResult(AbstractAggregationResult):
    cpu_integral: float
