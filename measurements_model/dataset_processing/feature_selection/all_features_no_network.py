import pandas as pd

from measurements_model.column_names import ProcessColumns, SystemColumns
from measurements_model.dataset_processing.feature_selection.feature_selector import FeatureSelector


class AllFeaturesNoNetwork(FeatureSelector):
    """
    Includes all process, hardware, idle and system features.
    Doesn't include energy consumption for idle and system.
    The system features are NOT a subtraction of idle and process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        network_system_columns = [SystemColumns.NETWORK_BYTES_SENT_SYSTEM_COL, SystemColumns.NETWORK_PACKETS_SENT_SYSTEM_COL,
                                  SystemColumns.NETWORK_BYTES_RECEIVED_SYSTEM_COL, SystemColumns.NETWORK_PACKETS_RECEIVED_SYSTEM_COL]
        network_process_columns = [ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL, ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL,
                                ProcessColumns.NETWORK_PACKETS_RECEIVED_PROCESS_COL, ProcessColumns.NETWORK_PACKETS_SENT_PROCESS_COL]
        return df.drop(network_system_columns + network_process_columns, axis=1, errors='ignore')
