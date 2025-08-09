from dataclasses import dataclass

from aggregative_results.raw_results_dtos import ProcessRawResults, SystemRawResults
from aggregative_results.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class SystemProcessRawResults(AbstractRawResults):
    processes_raw_results: ProcessRawResults
    system_raw_results: SystemRawResults
