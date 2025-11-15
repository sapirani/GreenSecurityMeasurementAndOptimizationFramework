import pandas as pd

from energy_model.configs.columns import SystemColumns
from measurements_model_pipeline.dataset_processing.feature_selection.feature_selector import FeatureSelector


class ProcessAndHardwareFeatureSelector(FeatureSelector):
    """
    Includes all process and hardware features.
    Doesn't include any idle or system features.
    Includes the energy consumption of a process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.CPU_SYSTEM_COL,
                        SystemColumns.MEMORY_SYSTEM_COL,
                        SystemColumns.DISK_READ_BYTES_SYSTEM_COL,
                        SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_READ_TIME,
                        SystemColumns.NETWORK_BYTES_SENT_SYSTEM_COL,
                        SystemColumns.NETWORK_PACKETS_SENT_SYSTEM_COL,
                        SystemColumns.NETWORK_BYTES_RECEIVED_SYSTEM_COL,
                        SystemColumns.NETWORK_PACKETS_RECEIVED_SYSTEM_COL,
                        SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL,
                        SystemColumns.DURATION_COL,
                        SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL,
                        SystemColumns.BATCH_ID_COL,
                        SystemColumns.SESSION_ID_COL],
                       axis=1, errors="ignore")
