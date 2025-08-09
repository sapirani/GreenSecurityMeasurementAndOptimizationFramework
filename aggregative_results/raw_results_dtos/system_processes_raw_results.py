from dataclasses import dataclass
from typing import List

from aggregative_results.raw_results_dtos import ProcessRawResults, SystemRawResults
from aggregative_results.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class SystemProcessesRawResults(AbstractRawResults):
    desired_process_raw_results: ProcessRawResults
    processes_raw_results: List[ProcessRawResults]
    system_raw_results: SystemRawResults
