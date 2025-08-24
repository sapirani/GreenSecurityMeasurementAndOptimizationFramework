import os

from aggregative_results.DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import \
    HardwareEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.process_extractor import ProcessExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.native_summary_extractor import \
    NativeSummaryExtractor

ALL_PROCESSES_CSV = fr"processes_data.csv"
SUMMARY_CSV = fr"summary.xlsx"
NETWORK_IO_PER_TIMESTAMP_CSV = fr"network_io_each_moment.csv"
HARDWARE_DETAILS_CSV = fr""
DEFAULT_SUMMARY_EXTRACTOR = NativeSummaryExtractor()

class MeasurementExtractor:
    def __init__(self, measurement_dir: str, summary_extractor: AbstractSummaryExtractor = DEFAULT_SUMMARY_EXTRACTOR):
        self.__measurement_dir = measurement_dir
        self.__summary_extractor = summary_extractor
        self.__process_extractor = ProcessExtractor()
        self.__hardware_extractor = HardwareExtractor()

    def extract_system_features(self) -> SystemEnergyModelFeatures:
        return self.__summary_extractor.extract_system_data(os.path.join(self.__measurement_dir, SUMMARY_CSV))

    def extract_process_features(self, process_name: str) -> ProcessEnergyModelFeatures:
        return self.__process_extractor.extract(os.path.join(self.__measurement_dir, ALL_PROCESSES_CSV), process_name)

    def extract_hardware_features(self) -> HardwareEnergyModelFeatures:
        return self.__hardware_extractor.extract(os.path.join(self.__measurement_dir, HARDWARE_DETAILS_CSV))