from abc import ABC, abstractmethod

from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures


class AbstractSummaryExtractor(ABC):

    @abstractmethod
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        pass
