import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_processing.feature_selection.system_only_feature_selector import SystemOnlyFeatureSelector
from energy_model.models.abstract_energy_model import AbstractEnergyModel


class SystemEnergyModel(AbstractEnergyModel):
    def __init__(self):
        super().__init__()

    def build_energy_model(self, df: pd.DataFrame):
        # filter irrelevant rows
        full_df_processed = self._full_df_filter_manager.filter_dataset(df)

        # Train a model on system columns only
        system_only_df = SystemOnlyFeatureSelector().select_features(full_df_processed)
        system_model, system_scaler = self._run_pipeline_executor(system_only_df, SystemColumns.ENERGY_USAGE_SYSTEM_COL)

        self._model = system_model
        self._scaler = system_scaler
