from dataclasses import dataclass
from typing import List

from aggregative_results.dtos import ProcessMetadata, AggregationResult


@dataclass
class AggregatedProcessResults:
    process_metadata: ProcessMetadata
    aggregation_results: List[AggregationResult]

    def merge(self, other: 'AggregatedProcessResults') -> None:
        if self.process_metadata != other.process_metadata:
            raise ValueError("Failed to merge process aggregation results, data belongs to a different process.")

        self.aggregation_results.extend(other.aggregation_results)
