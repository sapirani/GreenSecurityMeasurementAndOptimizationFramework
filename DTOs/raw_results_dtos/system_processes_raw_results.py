from dataclasses import dataclass
from typing import List

from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class FullScopeRawResults(AbstractRawResults):
    desired_process_raw_results: ProcessRawResults
    processes_raw_results: List[ProcessRawResults]
    system_raw_results: SystemRawResults
