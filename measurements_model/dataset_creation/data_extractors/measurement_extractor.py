import os

import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import \
    HardwareEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import TOTAL_CPU_CSV, TOTAL_MEMORY_CSV, DISK_IO_PER_TIMESTAMP_CSV, \
    NETWORK_IO_PER_TIMESTAMP_CSV, BATTERY_STATUS_CSV, ALL_PROCESSES_CSV
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.process_extractor import ProcessExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.system_resources_isolation_summary_extractor import \
    SystemResourcesIsolationSummaryExtractor
from utils.general_consts import CPUColumns, MemoryColumns, BatteryColumns, ProcessesColumns

DEFAULT_SUMMARY_EXTRACTOR = SystemResourcesIsolationSummaryExtractor()


class MeasurementExtractor:
    def __init__(self, measurement_dir: str, summary_extractor: AbstractSummaryExtractor = DEFAULT_SUMMARY_EXTRACTOR):
        self.__measurement_dir = measurement_dir

    def extract_total_cpu_usage(self) -> pd.DataFrame:
        total_cpu_file = os.path.join(self.__measurement_dir, TOTAL_CPU_CSV)
        df = pd.read_csv(total_cpu_file)
        relevant_df = df[[CPUColumns.TIME, CPUColumns.SUM_ACROSS_CORES_PERCENT]]
        return relevant_df

    def extract_total_memory_usage(self) -> pd.DataFrame:
        total_memory_file = os.path.join(self.__measurement_dir, TOTAL_MEMORY_CSV)
        df = pd.read_csv(total_memory_file)
        relevant_df = df[[MemoryColumns.TIME, MemoryColumns.USED_MEMORY]]
        return relevant_df

    def extract_total_disk_usage(self) -> pd.DataFrame:
        total_disk_file = os.path.join(self.__measurement_dir, DISK_IO_PER_TIMESTAMP_CSV)
        df = pd.read_csv(total_disk_file)
        return df

    def extract_total_network_usage(self) -> pd.DataFrame:
        total_network_file = os.path.join(self.__measurement_dir, NETWORK_IO_PER_TIMESTAMP_CSV)
        df = pd.read_csv(total_network_file)
        return df

    def extract_total_battery_usage(self) -> pd.DataFrame:
        total_battery_file = os.path.join(self.__measurement_dir, BATTERY_STATUS_CSV)
        df = pd.read_csv(total_battery_file)
        relevant_df = df[[BatteryColumns.TIME, BatteryColumns.VOLTAGE]]
        return relevant_df

    def extract_process_usage(self) -> pd.DataFrame:
        process_file = os.path.join(self.__measurement_dir, ALL_PROCESSES_CSV)
        df = pd.read_csv(process_file)
        process_of_interest_df = df[df[ProcessesColumns.PROCESS_OF_INTEREST] is True]
        return process_of_interest_df