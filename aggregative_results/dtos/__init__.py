from dataclasses import dataclass
from typing import Optional, List

from aggregative_results.dtos.aggregated_results_dtos import AggregationResult
from aggregative_results.dtos.raw_results_dtos import ProcessRawResults


@dataclass(frozen=True)
class ProcessIdentity:
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
