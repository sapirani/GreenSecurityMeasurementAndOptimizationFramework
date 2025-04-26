import pandas as pd

from measurements_model.config import SystemColumns, IDLEColumns
from measurements_model.dataset_processing.process_data.feature_selection.feature_selector import FeatureSelector


class ProcessAndFullSystem(FeatureSelector):  # no subtraction in system column
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL,
                        IDLEColumns.CPU_IDLE_COL, IDLEColumns.MEMORY_IDLE_COL, IDLEColumns.PAGE_FAULT_IDLE_COL,
                        IDLEColumns.DISK_READ_BYTES_IDLE_COL, IDLEColumns.DISK_READ_COUNT_IDLE_COL,
                        IDLEColumns.DISK_READ_TIME, IDLEColumns.DISK_WRITE_TIME, IDLEColumns.DISK_WRITE_BYTES_IDLE_COL,
                        IDLEColumns.DISK_WRITE_COUNT_IDLE_COL, IDLEColumns.DURATION_COL],
                       axis=1)
