from dataclasses import dataclass

from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from aggregative_results.DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class ProcessSystemRawResults(AbstractRawResults):
    process_raw_results: ProcessRawResults
    system_raw_results: SystemRawResults
