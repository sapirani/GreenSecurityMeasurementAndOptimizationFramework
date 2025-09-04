from dataclasses import dataclass
from typing import Optional, List

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults


@dataclass(frozen=True)
class ProcessIdentity:
    """
    Assuming pid and process_name are unique
    """
    pid: int
    process_name: str

    @staticmethod
    def from_raw_results(raw_results: ProcessRawResults) -> 'ProcessIdentity':
        return ProcessIdentity(
            pid=raw_results.pid,
            process_name=raw_results.process_name
        )


@dataclass
class ProcessMetadata:
    process_of_interest: bool = False
    arguments: Optional[List[AbstractAggregationResult]] = None
