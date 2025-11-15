import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_processing.feature_selection.process_and_hardware_feature_selector import \
    ProcessAndHardwareFeatureSelector
from energy_model.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector


class ProcessOnlyFeatureSelector(ProcessAndSystemFeatureSelector, ProcessAndHardwareFeatureSelector):
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_only_process = super().select_features(df)
        if SystemColumns.ENERGY_USAGE_SYSTEM_COL in df_with_only_process.columns:
            df_with_only_process = df_with_only_process.drop(SystemColumns.ENERGY_USAGE_SYSTEM_COL, axis=1)

        return df_with_only_process