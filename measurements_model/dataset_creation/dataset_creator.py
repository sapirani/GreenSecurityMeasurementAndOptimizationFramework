import glob
import os
from dataclasses import asdict

import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.full_energy_model_features import \
    EnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.config import DEFAULT_HARDWARE_FILE_PATH, TIME_COLUMN_NAME, ProcessColumns, SystemColumns
from measurements_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from measurements_model.dataset_creation.data_extractors.idle_extractor import IdleExtractor
from utils.general_consts import ProcessesColumns, NetworkIOColumns, DiskIOColumns, MemoryColumns, BatteryColumns, \
    CPUColumns

DEFAULT_TIME_PER_BATCH = 150
SESSIONS_DATAFRAMES_FILE_TYPE = "*.csv"


class DatasetCreator:
    def __init__(self, system_sessions_dir: str, idle_session_path: str):
        self.__system_sessions_dir = system_sessions_dir
        self.__idle_details = IdleExtractor().extract(idle_session_path)
        self.__hardware_details = HardwareExtractor().extract(DEFAULT_HARDWARE_FILE_PATH)

    def __extend_df(self, df: pd.DataFrame, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> pd.DataFrame:
        expended_df = self.__expend_df_with_relative_cpu(df)
        expended_df = self.__expend_df_with_duration(expended_df)
        expended_df = self.__expend_df_with_energy(expended_df, time_per_batch)
        return expended_df

    def __expend_df_with_relative_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SystemColumns.CPU_SYSTEM_COL] = df[CPUColumns.SUM_ACROSS_CORES_PERCENT].diff().fillna(0)
        df[ProcessColumns.CPU_PROCESS_COL] = df[ProcessesColumns.CPU_SUM_ACROSS_CORES].diff().fillna(0)

        return df

    def __expend_df_with_duration(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = df[SystemColumns.DURATION_COL] * df["energy_per_sec"] - df[SystemColumns.DURATION_COL] * self.__idle_details.energy_per_second
        return df

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
            number_of_page_faults_system=0,  # todo: change
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

    def __extract_session_features(self, session_file_path: str, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> \
            list[EnergyModelFeatures]:
        df = pd.read_csv(session_file_path)
        extended_df = self.__extend_df(df, time_per_batch)
        session_samples = []
        for _, row in extended_df.iterrows():
            timestamp = row[TIME_COLUMN_NAME]
            system_features = self.__extract_system_features(row)
            process_features = self.__extract_process_features(row)
            sample_features = EnergyModelFeatures(
                timestamp=timestamp,
                system_features=system_features,
                process_features=process_features
            )
            session_samples.append(sample_features)

        return session_samples

    def __read_sessions(self, time_per_batch: int = DEFAULT_TIME_PER_BATCH) -> list[EnergyModelFeatures]:
        sessions_files = glob.glob(os.path.join(self.__system_sessions_dir, SESSIONS_DATAFRAMES_FILE_TYPE))
        all_sessions = []
        for session_file in sessions_files:
            session_features = self.__extract_session_features(session_file, time_per_batch)
            all_sessions.extend(session_features)
        return all_sessions


    def __convert_sessions_to_df(self, all_sessions: list[EnergyModelFeatures]) -> pd.DataFrame:
        samples = [
            {**asdict(sample.process_features), **asdict(sample.system_features)}
            for sample in all_sessions
        ]
        return pd.DataFrame(samples)

    def create_dataset(self) -> pd.DataFrame:
        sessions_stats = self.__read_sessions()
        return self.__convert_sessions_to_df(sessions_stats)
