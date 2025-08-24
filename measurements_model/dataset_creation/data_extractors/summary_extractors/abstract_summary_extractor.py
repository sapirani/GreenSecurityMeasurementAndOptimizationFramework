from abc import ABC, abstractmethod

from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures


class AbstractSummaryExtractor(ABC):

    @abstractmethod
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        pass

    def extract_process_data(self, summary_file_path: str) -> ProcessEnergyModelFeatures:
        #todo: change and implement
        raise RuntimeError("Not implemented extracting process data from summary file")
