from dataclasses import dataclass
from typing import Optional, List

from aggregative_results.DTOs.aggregated_results_dtos import AggregationResult
from aggregative_results.DTOs.raw_results_dtos import ProcessRawResults


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
    arguments: Optional[List[AggregationResult]] = None
