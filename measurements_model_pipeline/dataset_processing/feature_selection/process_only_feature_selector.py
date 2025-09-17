import pandas as pd

from measurements_model_pipeline.dataset_processing.feature_selection.process_and_hardware_feature_selector import \
    ProcessAndHardwareFeatureSelector
from measurements_model_pipeline.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector


class ProcessOnlyFeatureSelector(ProcessAndSystemFeatureSelector, ProcessAndHardwareFeatureSelector):
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_only_process = super().select_features(df)
        return df_with_only_process