from dataclasses import asdict
from pathlib import Path
import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from measurements_model.config import ProcessColumns, NO_ENERGY_MEASURED
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from measurements_model.dataset_creation.data_extractors.measurement_extractor import MeasurementExtractor
from measurements_model.dataset_creation.data_extractors.summary_extractors.system_resources_isolation_summary_extractor import \
    SystemResourcesIsolationSummaryExtractor
from measurements_model.model_execution.main_model_configuration import PROCESS_NAME


class DatasetCreator:
    def __init__(self, idle_dir_path: str, measurements_dir_path: str):
        self.__idle_dir_path = idle_dir_path
        self.__measurements_dir_path = measurements_dir_path
        self.__idle_extractor = IdleExtractor()

    def __read_idle_stats(self) -> IdleEnergyModelFeatures:
        idle_results = self.__idle_extractor.extract(self.__idle_dir_path)
        return idle_results

    def __extract_sample(self, measurement_dir: str, idle_results: IdleEnergyModelFeatures) -> pd.Series:
        measurement_extractor = MeasurementExtractor(measurement_dir=measurement_dir)
        system_summary_results = measurement_extractor.extract_system_features()
        process_summary_results = measurement_extractor.extract_process_features(process_name=PROCESS_NAME)
        hardware_results = measurement_extractor.extract_hardware_features()

        system_total_energy_consumption = system_summary_results.total_energy_consumption_system_mWh
        idle_energy_consumption = idle_results.total_energy_consumption_in_idle_mWh

        process_energy_value = system_total_energy_consumption - idle_energy_consumption if system_total_energy_consumption > 0 else NO_ENERGY_MEASURED
        new_sample = {**asdict(process_summary_results), **asdict(system_summary_results), **asdict(idle_results), **asdict(hardware_results),
                      ProcessColumns.ENERGY_USAGE_PROCESS_COL: process_energy_value}

        new_sample = {key: value for key, value in new_sample.items() if value is not None}
        return pd.Series(new_sample)

    def __read_measurements(self) -> pd.DataFrame:
        idle_results = self.__read_idle_stats()
        df_rows = []
        for measurement_dir in Path(self.__measurements_dir_path).iterdir():
            if measurement_dir.is_dir():
                print("Collecting info from " + measurement_dir.name)
                samples = self.__extract_samples(measurement_dir=measurement_dir)
                new_sample = self.__extract_sample(measurement_dir=measurement_dir, idle_results=idle_results)
                df_rows.append(new_sample)

        return pd.DataFrame(df_rows)

    def create_dataset(self) -> pd.DataFrame:
        df = self.__read_measurements()
        return df
