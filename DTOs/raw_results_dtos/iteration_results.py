from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from consts import ElasticIndex


@dataclass
class IterationResults:
    system_result: Optional[SystemRawResults] = None
    process_results: List[ProcessRawResults] = field(default_factory=list)

    def add_result(self, index: ElasticIndex, raw_results: Dict[str, Any]):
        if index == ElasticIndex.SYSTEM:
            self.__set_system_result(SystemRawResults.from_dict(raw_results))
        elif index == ElasticIndex.PROCESS:
            self.__add_process_result(ProcessRawResults.from_dict(raw_results))
        else:
            raise ValueError("Received unexpected index")

    def __set_system_result(self, result: SystemRawResults):
        self.system_result = result

    def __add_process_result(self, result: ProcessRawResults):
        self.process_results.append(result)

    def get_system_result(self) -> SystemRawResults:
        return self.system_result

    def get_processes_results(self) -> List[ProcessRawResults]:
        return self.process_results
