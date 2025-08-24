from dataclasses import asdict
from functools import reduce
from pathlib import Path
import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.full_energy_model_features import \
    EnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import \
    HardwareEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import DEFAULT_HARDWARE_FILE_PATH
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from measurements_model.dataset_creation.data_extractors.measurement_extractor import MeasurementExtractor
from measurements_model.dataset_creation.data_extractors.utils import merge_dfs


class DatasetCreator:
    def __init__(self, idle_dir_path: str, measurements_dir_path: str):
        self.__idle_dir_path = idle_dir_path
        self.__measurements_dir_path = measurements_dir_path
        self.__idle_extractor = IdleExtractor()
        self.__hardware_extractor = HardwareExtractor()

    def __read_idle_stats(self) -> IdleEnergyModelFeatures:
        return self.__idle_extractor.extract(self.__idle_dir_path)

    def __read_hardware_stats(self) -> HardwareEnergyModelFeatures:
        return self.__hardware_extractor.extract(DEFAULT_HARDWARE_FILE_PATH)

    def __read_all_usage(self, measurement_dir: str) -> pd.DataFrame:
        measurement_extractor = MeasurementExtractor(measurement_dir=measurement_dir)
        total_cpu = measurement_extractor.extract_total_cpu_usage()
        total_memory = measurement_extractor.extract_total_memory_usage()
        total_disk = measurement_extractor.extract_total_disk_usage()
        total_network = measurement_extractor.extract_total_network_usage()
        total_battery = measurement_extractor.extract_total_battery_usage()
        process_usage = measurement_extractor.extract_process_usage()

        all_dfs = [total_cpu, total_memory, total_disk, total_network, total_battery, process_usage]
        full_df = reduce(merge_dfs, all_dfs)
        return full_df

    def __extract_samples_from_measurement(self, measurement_dir: str) -> list[EnergyModelFeatures]:
        full_usage_df = self.__read_all_usage(measurement_dir)

    # def __extract_sample(self, measurement_dir: str, idle_results: IdleEnergyModelFeatures) -> pd.Series:
    #     measurement_extractor = MeasurementExtractor(measurement_dir=measurement_dir)
    #     system_summary_results = measurement_extractor.extract_system_features()
    #     process_summary_results = measurement_extractor.extract_process_features(process_name=PROCESS_NAME)
    #     hardware_results = measurement_extractor.extract_hardware_features()
    #
    #     system_total_energy_consumption = system_summary_results.total_energy_consumption_system_mWh
    #     idle_energy_consumption = idle_results.total_energy_consumption_in_idle_mWh
    #
    #     process_energy_value = system_total_energy_consumption - idle_energy_consumption if system_total_energy_consumption > 0 else NO_ENERGY_MEASURED
    #     new_sample = {**asdict(process_summary_results), **asdict(system_summary_results), **asdict(idle_results), **asdict(hardware_results),
    #                   ProcessColumns.ENERGY_USAGE_PROCESS_COL: process_energy_value}
    #
    #     new_sample = {key: value for key, value in new_sample.items() if value is not None}
    #     return pd.Series(new_sample)

    def __read_measurements(self) -> pd.DataFrame:
        idle_results = self.__read_idle_stats()
        hardware_results = self.__read_hardware_stats()
        df_rows = []
        for measurement_dir in Path(self.__measurements_dir_path).iterdir():
            if measurement_dir.is_dir():
                print("Collecting info from " + measurement_dir.name)
                samples = self.__extract_samples_from_measurement(measurement_dir=measurement_dir)
                df_rows.extend(samples)

        df_no_energy_no_hardware = pd.DataFrame(df_rows)
        # df_with_idle_energy = df_no_energy_no_hardware.copy()
        # df_with_idle_energy[IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL] = df_no_energy_no_hardware[SystemColumns.DURATION_COL] * idle_results.energy_per_second
        # full_df = df_with_idle_energy.join(pd.DataFrame(asdict(hardware_results), index=df_with_idle_energy.index))
        #
        # return full_df
        return df_no_energy_no_hardware

    def create_dataset(self) -> pd.DataFrame:
        df = self.__read_measurements()
        return df
