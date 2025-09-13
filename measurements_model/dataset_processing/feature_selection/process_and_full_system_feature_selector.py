import pandas as pd

from measurements_model.config import TIME_COLUMN_NAME
from measurements_model.column_names import SystemColumns, IDLEColumns
from measurements_model.dataset_processing.feature_selection.feature_selector import FeatureSelector


class ProcessAndTotalSystem(FeatureSelector):
    """
    Includes all process, hardware and system features.
    Doesn't include any idle features.
    Includes the energy consumption of a process BUT not the total energy consumption of the system.
    The system features are NOT a subtraction of idle and process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL,
                        SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL,
                        SystemColumns.BATCH_ID_COLUMN,
                        TIME_COLUMN_NAME,
                        IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL,
                        IDLEColumns.CPU_IDLE_COL, IDLEColumns.MEMORY_IDLE_COL, IDLEColumns.PAGE_FAULT_IDLE_COL,
                        IDLEColumns.DISK_READ_BYTES_IDLE_COL, IDLEColumns.DISK_READ_COUNT_IDLE_COL,
                        IDLEColumns.DISK_READ_TIME, IDLEColumns.DISK_WRITE_TIME, IDLEColumns.DISK_WRITE_BYTES_IDLE_COL,
                        IDLEColumns.DISK_WRITE_COUNT_IDLE_COL, IDLEColumns.DURATION_COL],
                       axis=1, errors='ignore')
