import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_processing.feature_selection.process_and_hardware_feature_selector import \
    ProcessAndHardwareFeatureSelector
from energy_model.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector


class ProcessOnlyFeatureSelector(ProcessAndSystemFeatureSelector, ProcessAndHardwareFeatureSelector):
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_only_process = super().select_features(df)
        df_with_only_process_without_system_energy = df_with_only_process.drop(SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                                                                               axis=1, errors='ignore')

        return df_with_only_process_without_system_energy
