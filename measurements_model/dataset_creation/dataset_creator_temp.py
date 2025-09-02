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
from measurements_model.config import DEFAULT_HARDWARE_FILE_PATH, SystemColumns, ProcessColumns
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from measurements_model.dataset_creation.data_extractors.measurement_extractor import MeasurementExtractor
from measurements_model.dataset_creation.data_extractors.utils import merge_dfs
from utils.general_consts import CPUColumns, ProcessesColumns, BatteryColumns, MemoryColumns, DiskIOColumns, \
    NetworkIOColumns

DEFAULT_TIME_PER_BATCH = 150

# todo: remove when moving permanently to the new approach.
class DatasetCreatorTemp:
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

    def __extract_samples_from_measurement(self, measurement_dir: str, idle_energy_per_sec: float, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> list[EnergyModelFeatures]:
        full_usage_df = self.__read_all_usage(measurement_dir)
        expended_df = self.__expend_df_with_relative_cpu(full_usage_df)
        expended_df = self.__expend_df_with_duration(expended_df)
        expended_df = self.__expend_df_with_energy(expended_df, time_per_batch, idle_energy_per_sec)
        return self.__convert_df_to_objects(expended_df)

    def __expend_df_with_relative_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.CPU_SYSTEM_COL] = df[CPUColumns.SUM_ACROSS_CORES_PERCENT].diff().fillna(0)
        df[ProcessColumns.CPU_PROCESS_COL] = df[ProcessesColumns.CPU_SUM_ACROSS_CORES].diff().fillna(0)

        return df

    def __expend_df_with_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.DURATION_COL] = df[CPUColumns.TIME].diff().fillna(0)
        return df

    def __expend_df_with_energy(self, df: pd.DataFrame, time_per_batch: int, idle_energy_per_sec: float) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL] = df[BatteryColumns.CAPACITY].diff().fillna(0)
        df["batch_id"] = (df[CPUColumns.TIME] // time_per_batch).astype(int)

        # Step 4: compute energy usage per second per batch
        energy_per_batch = (
            df.groupby("batch_id")[BatteryColumns.CAPACITY]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / time_per_batch)
            .rename("energy_per_sec")
        )
        df = df.merge(energy_per_batch, on="batch_id", how="left")

        # Step 5: energy usage per row
        df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = df[SystemColumns.DURATION_COL] * df["energy_per_sec"] - df[SystemColumns.DURATION_COL] * idle_energy_per_sec
        return df

    def __convert_df_to_objects(self, df: pd.DataFrame) -> list[EnergyModelFeatures]:
        all_rows = []
        for _, row in df.iterrows():
            timestamp = row[CPUColumns.TIME]
            system_features = self.__extract_system_features(row)
            process_features = self.__extract_process_features(row)
            full_features = EnergyModelFeatures(timestamp=timestamp, system_features=system_features, process_features=process_features, hardware_features=None, idle_features=None)
            all_rows.append(full_features)

        return all_rows

    def __extract_system_features(self, data_row: pd.Series) -> SystemEnergyModelFeatures:
        return SystemEnergyModelFeatures(
            cpu_usage_system=data_row[SystemColumns.CPU_SYSTEM_COL],
            memory_gb_usage_system=data_row[MemoryColumns.USED_MEMORY],
            disk_read_bytes_kb_usage_system=data_row[DiskIOColumns.READ_BYTES],
            disk_read_count_usage_system=data_row[DiskIOColumns.READ_COUNT],
            disk_write_bytes_kb_usage_system=data_row[DiskIOColumns.WRITE_BYTES],
            disk_write_count_usage_system=data_row[DiskIOColumns.WRITE_COUNT],
            disk_read_time_system_ms_sum=data_row[DiskIOColumns.READ_TIME],
            disk_write_time_system_ms_sum=data_row[DiskIOColumns.WRITE_TIME],
            number_of_page_faults_system=0, #todo: change
            network_bytes_kb_sum_sent_system=data_row[NetworkIOColumns.KB_SENT],
            network_packets_sum_sent_system=data_row[NetworkIOColumns.PACKETS_SENT],
            network_bytes_kb_sum_received_system=data_row[NetworkIOColumns.KB_RECEIVED],
            network_packets_sum_received_system=data_row[NetworkIOColumns.PACKETS_RECEIVED],
            total_energy_consumption_system_mWh=data_row[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL]
        )

    def __extract_process_features(self, data_row: pd.Series) -> ProcessEnergyModelFeatures:
        return ProcessEnergyModelFeatures(
            cpu_usage_process=data_row[ProcessesColumns.CPU_SUM_ACROSS_CORES],
            memory_mb_usage_process=data_row[ProcessesColumns.USED_MEMORY],
            disk_read_bytes_kb_usage_process=data_row[ProcessesColumns.READ_BYTES],
            disk_read_count_usage_process=data_row[ProcessesColumns.READ_COUNT],
            disk_write_bytes_kb_usage_process=data_row[ProcessesColumns.WRITE_BYTES],
            disk_write_count_usage_process=data_row[ProcessesColumns.WRITE_COUNT],
            network_bytes_sum_kb_sent_process=data_row[ProcessesColumns.BYTES_SENT],
            network_packets_sum_sent_process=data_row[ProcessesColumns.PACKETS_SENT],
            network_bytes_sum_kb_received_process=data_row[ProcessesColumns.BYTES_RECEIVED],
            network_packets_sum_received_process=data_row[ProcessesColumns.PACKETS_RECEIVED],
            energy_consumption_process_mWh=data_row[ProcessColumns.ENERGY_USAGE_PROCESS_COL])



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
                samples_as_obj = self.__extract_samples_from_measurement(measurement_dir=measurement_dir,
                                                                  idle_energy_per_sec=idle_results.energy_per_second,
                                                                  time_per_batch=DEFAULT_TIME_PER_BATCH)

                samples = [
                    {**asdict(sample.process_features), **asdict(sample.system_features)}
                    for sample in samples_as_obj
                ]
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
