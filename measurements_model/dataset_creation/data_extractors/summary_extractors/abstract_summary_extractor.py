from abc import ABC, abstractmethod

import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures
from measurements_model.dataset_creation.data_extractors.summary_extractors.summary_columns import SummaryColumns


class AbstractSummaryExtractor(ABC):

    @abstractmethod
    def extract_system_data(self, summary_file_path: str) -> SystemEnergyModelFeatures:
        pass

    def _extract_data_from_df(self, df: pd.DataFrame, columns: SummaryColumns) -> SystemEnergyModelFeatures:
        total_column = columns.total_column

        network_sent_bytes = None
        network_sent_packets = None
        network_received_bytes = None
        network_received_packets = None
        if columns.network_bytes_kb_sum_sent_column is not None:
            network_sent_bytes = df.loc[columns.network_bytes_kb_sum_sent_column, total_column]

        if columns.network_packets_sum_sent_column is not None:
            network_sent_packets = df.loc[columns.network_packets_sum_sent_column, total_column]

        if columns.network_bytes_kb_sum_received_column is not None:
            network_received_bytes = df.loc[columns.network_bytes_kb_sum_received_column, total_column]

        if columns.network_packets_sum_received_column is not None:
            network_received_packets = df.loc[columns.network_packets_sum_received_column, total_column]

        return SystemEnergyModelFeatures(
            cpu_usage_system=df.loc[columns.cpu_column, total_column],
            memory_gb_usage_system=df.loc[columns.memory_column, total_column],
            disk_read_bytes_kb_usage_system=df.loc[columns.disk_read_bytes_column, total_column],
            disk_read_count_usage_system=df.loc[columns.disk_read_count_column, total_column],
            disk_write_bytes_kb_usage_system=df.loc[columns.disk_write_bytes_column, total_column],
            disk_write_count_usage_system=df.loc[columns.disk_write_count_column, total_column],
            disk_read_time_system_ms_sum=df.loc[columns.disk_read_time_column, total_column],
            disk_write_time_system_ms_sum=df.loc[columns.disk_write_time_column, total_column],
            network_bytes_kb_sum_sent_system=network_sent_bytes,
            network_packets_sum_sent_system=network_sent_packets,
            network_bytes_kb_sum_received_system=network_received_bytes,
            network_packets_sum_received_system=network_received_packets,
            number_of_page_faults_system=df.loc[columns.number_of_page_faults_column, total_column],
            total_energy_consumption_system_mWh=df.loc[columns.total_energy_consumption_column, total_column]
        )


    def extract_process_data(self, summary_file_path: str) -> ProcessEnergyModelFeatures:
        #todo: change and implement
        raise RuntimeError("Not implemented extracting process data from summary file")
