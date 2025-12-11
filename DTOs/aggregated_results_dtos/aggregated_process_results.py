from dataclasses import dataclass
from typing import Dict

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from DTOs.process_info import ProcessMetadata
from elastic_reader.aggregators.aggregation_types import AggregationType


@dataclass
class AggregatedProcessResults:
    process_metadata: ProcessMetadata
    aggregation_results: Dict[AggregationType, AbstractAggregationResult]

    def merge(self, other: 'AggregatedProcessResults') -> None:
        if self.process_metadata != other.process_metadata:
            raise ValueError("Failed to merge process aggregation results, data belongs to a different process.")

        overlapping_aggregations = other.aggregation_results.keys() & self.aggregation_results.keys()
        if overlapping_aggregations:
            raise ValueError("Cannot merge process results with the same aggregation type. "
                             f"Overlapping aggregations: {overlapping_aggregations}.")

        self.aggregation_results.update(other.aggregation_results)
