from dataclasses import asdict

import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.full_energy_model_features import \
    EnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import DEFAULT_HARDWARE_FILE_PATH, TIME_COLUMN_NAME, ProcessColumns, SystemColumns, \
    PROCESS_COLUMN_SUFFIX, SYSTEM_COLUMN_SUFFIX
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from measurements_model.dataset_creation.dataframe_utils import get_full_features_dataframe
from utils.general_consts import ProcessesColumns, NetworkIOColumns, DiskIOColumns, MemoryColumns, BatteryColumns, \
    CPUColumns

DEFAULT_TIME_PER_BATCH = 150


class DatasetCreator:
    def __init__(self, idle_session_path: str):
        self.__idle_details = IdleExtractor().extract(idle_session_path)
        self.__hardware_details = HardwareExtractor().extract(DEFAULT_HARDWARE_FILE_PATH)

    def __extend_df(self, df: pd.DataFrame, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> pd.DataFrame:
        expended_df = DatasetCreator.__expend_df_with_relative_cpu(df)
        expended_df = DatasetCreator.__expend_df_with_duration(expended_df)
        expended_df = self.__expend_df_with_energy(expended_df, time_per_batch)
        expended_df = self.__expend_df_with_hardware(expended_df)
        return expended_df

    @staticmethod
    def __expend_df_with_relative_cpu(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.CPU_SYSTEM_COL] = df[f"{CPUColumns.SUM_ACROSS_CORES_PERCENT.value}{SYSTEM_COLUMN_SUFFIX}"].diff().fillna(0)
        df[ProcessColumns.CPU_PROCESS_COL] = df[f"{ProcessesColumns.CPU_SUM_ACROSS_CORES.value}{PROCESS_COLUMN_SUFFIX}"].diff().fillna(0)

        return df

    @staticmethod
    def __expend_df_with_duration(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.DURATION_COL] = df[TIME_COLUMN_NAME].diff().fillna(0)
        return df

    def __expend_df_with_energy(self, df: pd.DataFrame, time_per_batch: int) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL] = df[BatteryColumns.CAPACITY].diff().fillna(0)
        df["batch_id"] = (df[TIME_COLUMN_NAME] // time_per_batch).astype(int)

        # Step 4: compute energy usage per second per batch
        energy_per_batch = (
            df.groupby("batch_id")[BatteryColumns.CAPACITY]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / time_per_batch)
            .rename("energy_per_sec")
        )
        df = df.merge(energy_per_batch, on="batch_id", how="left")

        # Step 5: energy usage per row
        df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = df[SystemColumns.DURATION_COL] * df["energy_per_sec"] - df[
            SystemColumns.DURATION_COL] * self.__idle_details.energy_per_second
        return df

    def __expend_df_with_hardware(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num_of_rows = len(df)
        hardware_dict = asdict(self.__hardware_details)
        hardware_df = pd.concat([pd.DataFrame(hardware_dict)] * num_of_rows, ignore_index=True)
        df_with_hardware = pd.concat([df.reset_index(drop=True), hardware_df.reset_index(drop=True)], axis=1)

        return df_with_hardware

    def __extract_system_features(self, data_row: pd.Series) -> SystemEnergyModelFeatures:
        return SystemEnergyModelFeatures(
            cpu_usage_system=data_row[SystemColumns.CPU_SYSTEM_COL],
            memory_gb_usage_system=data_row[f"{MemoryColumns.USED_MEMORY.value}{SYSTEM_COLUMN_SUFFIX}"],
            disk_read_bytes_kb_usage_system=data_row[f"{DiskIOColumns.READ_BYTES.value}{SYSTEM_COLUMN_SUFFIX}"],
            disk_read_count_usage_system=data_row[f"{DiskIOColumns.READ_COUNT.value}{SYSTEM_COLUMN_SUFFIX}"],
            disk_write_bytes_kb_usage_system=data_row[f"{DiskIOColumns.WRITE_BYTES.value}{SYSTEM_COLUMN_SUFFIX}"],
            disk_write_count_usage_system=data_row[f"{DiskIOColumns.WRITE_COUNT.value}{SYSTEM_COLUMN_SUFFIX}"],
            disk_read_time_system_ms_sum=data_row[DiskIOColumns.READ_TIME],
            disk_write_time_system_ms_sum=data_row[DiskIOColumns.WRITE_TIME],
            number_of_page_faults_system=0,  # todo: change
            network_bytes_kb_sum_sent_system=data_row[f"{NetworkIOColumns.KB_SENT.value}{SYSTEM_COLUMN_SUFFIX}"],
            network_packets_sum_sent_system=data_row[f"{NetworkIOColumns.PACKETS_SENT.value}{SYSTEM_COLUMN_SUFFIX}"],
            network_bytes_kb_sum_received_system=data_row[f"{NetworkIOColumns.KB_RECEIVED.value}{SYSTEM_COLUMN_SUFFIX}"],
            network_packets_sum_received_system=data_row[f"{NetworkIOColumns.PACKETS_RECEIVED.value}{SYSTEM_COLUMN_SUFFIX}"],
            total_energy_consumption_system_mWh=data_row[SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL]
        )

    def __extract_process_features(self, data_row: pd.Series) -> ProcessEnergyModelFeatures:
        return ProcessEnergyModelFeatures(
            cpu_usage_process=data_row[ProcessColumns.CPU_PROCESS_COL],
            memory_mb_usage_process=data_row[f"{ProcessesColumns.USED_MEMORY.value}{PROCESS_COLUMN_SUFFIX}"],
            disk_read_bytes_kb_usage_process=data_row[f"{ProcessesColumns.READ_BYTES.value}{PROCESS_COLUMN_SUFFIX}"],
            disk_read_count_usage_process=data_row[f"{ProcessesColumns.READ_COUNT.value}{PROCESS_COLUMN_SUFFIX}"],
            disk_write_bytes_kb_usage_process=data_row[f"{ProcessesColumns.WRITE_BYTES.value}{PROCESS_COLUMN_SUFFIX}"],
            disk_write_count_usage_process=data_row[f"{ProcessesColumns.WRITE_COUNT.value}{PROCESS_COLUMN_SUFFIX}"],
            network_bytes_sum_kb_sent_process=data_row[f"{ProcessesColumns.BYTES_SENT.value}{PROCESS_COLUMN_SUFFIX}"],
            network_packets_sum_sent_process=data_row[f"{ProcessesColumns.PACKETS_SENT.value}{PROCESS_COLUMN_SUFFIX}"],
            network_bytes_sum_kb_received_process=data_row[f"{ProcessesColumns.BYTES_RECEIVED.value}{PROCESS_COLUMN_SUFFIX}"],
            network_packets_sum_received_process=data_row[f"{ProcessesColumns.PACKETS_RECEIVED.value}{PROCESS_COLUMN_SUFFIX}"],
            energy_consumption_process_mWh=data_row[ProcessColumns.ENERGY_USAGE_PROCESS_COL])

    def __read_data(self, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> list[EnergyModelFeatures]:
        df = get_full_features_dataframe()
        extended_df = self.__extend_df(df, time_per_batch)
        all_data = []
        for _, row in extended_df.iterrows():
            timestamp = row[TIME_COLUMN_NAME]
            system_features = self.__extract_system_features(row)
            process_features = self.__extract_process_features(row)
            sample_features = EnergyModelFeatures(
                timestamp=timestamp,
                system_features=system_features,
                process_features=process_features
            )
            all_data.append(sample_features)

        return all_data

    def __convert_objects_to_df(self, all_data: list[EnergyModelFeatures]) -> pd.DataFrame:
        samples = [
            {**asdict(sample.process_features), **asdict(sample.system_features)}
            for sample in all_data
        ]
        return pd.DataFrame(samples)

    def create_dataset(self) -> pd.DataFrame:
        data = self.__read_data()
        return self.__convert_objects_to_df(data)
