import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.configs.defaults_configs import BEST_SYSTEM_MODEL_METRIC
from energy_model.dataset_processing.feature_selection.system_only_feature_selector import SystemOnlyFeatureSelector
from energy_model.energy_model_parameters import SYSTEM_ONLY_DF_PATH
from energy_model.models.abstract_energy_model import AbstractEnergyModel


class SystemEnergyModel(AbstractEnergyModel):
    def __init__(self):
        super().__init__()

    def build_energy_model(self, df: pd.DataFrame):
        # filter irrelevant rows
        full_df_processed = self._full_df_filter_manager.filter_dataset(df)

        # Train a model on system columns only
        system_only_df = SystemOnlyFeatureSelector().select_features(full_df_processed)
        system_only_df.to_csv(SYSTEM_ONLY_DF_PATH)

        system_model, system_scaler = self._run_pipeline_executor(system_only_df, SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                                                                  best_model_metric_name=BEST_SYSTEM_MODEL_METRIC,
                                                                  hyper_parameters={
                                                                      "loss": "squared_error",
                                                                      "max_depth": 16,
                                                                      "min_samples_leaf": 10,
                                                                      "max_iter": 400,
                                                                      "l2_regularization": 0.1
                                                                  })

        self._model = system_model
        self._scaler = system_scaler
