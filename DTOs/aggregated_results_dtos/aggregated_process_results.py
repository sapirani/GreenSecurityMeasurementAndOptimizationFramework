from dataclasses import dataclass
from typing import List

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from DTOs.process_info import ProcessMetadata


@dataclass
class AggregatedProcessResults:
    process_metadata: ProcessMetadata
    aggregation_results: List[AbstractAggregationResult]

    def merge(self, other: 'AggregatedProcessResults') -> None:
        if self.process_metadata != other.process_metadata:
            raise ValueError("Failed to merge process aggregation results, data belongs to a different process.")

        self.aggregation_results.extend(other.aggregation_results)
