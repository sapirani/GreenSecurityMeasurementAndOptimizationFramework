import pandas as pd

from energy_model.configs.columns import ProcessColumns
from energy_model.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector


class SystemOnlyFeatureSelector(ProcessAndSystemFeatureSelector):
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_system_and_process = super().select_features(df)
        df_with_system_and_process_without_process_energy = df_with_system_and_process.drop(
            ProcessColumns.ENERGY_USAGE_PROCESS_COL, axis=1, errors='ignore')

        return df_with_system_and_process_without_process_energy.drop(
            [
                ProcessColumns.CPU_PROCESS_COL,
                ProcessColumns.MEMORY_PROCESS_COL,
                ProcessColumns.MEMORY_PROCESS_COL,
                ProcessColumns.DISK_READ_BYTES_PROCESS_COL,
                ProcessColumns.DISK_READ_COUNT_PROCESS_COL,
                ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL,
                ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL,
                ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL,
                ProcessColumns.NETWORK_PACKETS_RECEIVED_PROCESS_COL,
                ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL,
                ProcessColumns.NETWORK_PACKETS_SENT_PROCESS_COL,
                ProcessColumns.PAGE_FAULT_PROCESS_COL
            ], axis=1)
