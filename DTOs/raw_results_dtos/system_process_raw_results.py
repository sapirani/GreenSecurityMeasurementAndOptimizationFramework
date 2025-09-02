from dataclasses import dataclass

from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class ProcessSystemRawResults(AbstractRawResults):
    processes_raw_results: ProcessRawResults
    system_raw_results: SystemRawResults
