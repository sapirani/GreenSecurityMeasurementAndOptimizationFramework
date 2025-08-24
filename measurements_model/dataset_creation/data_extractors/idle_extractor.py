import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor


class IdleExtractor:
    def __init__(self, summary_extractor: AbstractSummaryExtractor):
        self.__summary_extractor = summary_extractor

    def extract(self, idle_summary_file_path: str) -> IdleEnergyModelFeatures:
        df = pd.read_csv(idle_summary_file_path)
        idle_summary_results = self.__summary_extractor.extract_system_data(df)
        return IdleEnergyModelFeatures()