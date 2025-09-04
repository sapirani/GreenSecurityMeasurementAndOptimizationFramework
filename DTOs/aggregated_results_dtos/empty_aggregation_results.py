from dataclasses import dataclass

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult


@dataclass
class EmptyAggregationResults(AbstractAggregationResult):
    pass
