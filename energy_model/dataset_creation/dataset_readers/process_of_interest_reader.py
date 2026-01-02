from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader


class ProcessOfInterestReader(DatasetReader):
    def _should_use_sample(self, system_raw_results: SystemRawResults, process_raw_results: ProcessRawResults,
                           iteration_metadata: IterationMetadata) -> bool:
        return process_raw_results.process_of_interest

    def get_name(self) -> str:
        return "process_of_interest_reader"