import pandas as pd

from measurements_model.config import SystemColumns
from measurements_model.dataset_processing.feature_selection.process_and_full_system_feature_selector import \
    ProcessAndTotalSystem


class ProcessAndHardware(ProcessAndTotalSystem):
    """
    Includes all process and hardware features.
    Doesn't include any idle or system features.
    Includes the energy consumption of a process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().select_features(df)
        return df.drop([SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                        SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME,
                        SystemColumns.PAGE_FAULT_SYSTEM_COL],
                       axis=1)
