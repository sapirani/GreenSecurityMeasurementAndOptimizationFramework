import pandas as pd

from energy_model.configs.columns import SystemColumns, ProcessColumns, COLUMNS_MAPPING
from energy_model.configs.defaults_configs import BEST_PROCESS_MODEL_METRIC
from energy_model.dataset_processing.filters.filter_manager import FilterManager
from energy_model.energy_model_parameters import PROCESS_SYSTEM_DF_PATH
from energy_model.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector
from energy_model.dataset_processing.feature_selection.process_only_feature_selector import ProcessOnlyFeatureSelector
from energy_model.dataset_processing.feature_selection.system_only_feature_selector import SystemOnlyFeatureSelector
from energy_model.dataset_processing.filters.energy_filter import EnergyFilter
from energy_model.models.abstract_energy_model import AbstractEnergyModel
from energy_model.models.system_energy_model import SystemEnergyModel


class ProcessEnergyModel(AbstractEnergyModel):
    def __init__(self, system_model: SystemEnergyModel):
        super().__init__()
        self.__system_model = system_model
        self.__process_energy_filter_manager = FilterManager(
            filters=[EnergyFilter(ProcessColumns.ENERGY_USAGE_PROCESS_COL)])

    def build_energy_model(self, full_df: pd.DataFrame):
        """
        Create a model that predicts energy usage based on process and system telemetry.
        Steps:
        1. Create system-features only dataframe.
        2. Create process-features only dataframe.
        3. Subtract process telemetry from system telemetry to receive system_without_proces dataframe.
        4. Predict energy usage of system-process using the pretrained system model.
        5. Subtract the prediction results from the given energy usage of system only to receive energy usage for process telemetry.
        6. Train new model using the process energy results as the target, and the original dataframe as the features.

        Input:
            full_df: Pandas DataFrame containing process and system telemetry.
        """
        if self._model is not None:
            raise Exception('This method should be called once!')

        # filter irrelevant rows
        full_df_processed = self._full_df_filter_manager.filter_dataset(full_df)

        process_energy_predictions = self.__get_process_energy(full_df_processed)
        process_system_df = self.__build_df_with_target(full_df_processed, process_energy_predictions)
        self.__train_process_energy_model(process_system_df)

    def __get_process_energy(self, full_df: pd.DataFrame) -> pd.Series:
        """
        Use the pretrained system model to predict energy usage of system-process telemetry.
        Subtract the given predictions from the total system's energy usage to receive process energy usage.
        (Perform steps 1-5 in the pipeline)
        """
        # Build System - Process dataset
        process_only_df = ProcessOnlyFeatureSelector().select_features(full_df)
        system_only_df = SystemOnlyFeatureSelector().select_features(full_df)
        system_no_process_df = self.__build_df_without_process(system_only_df, process_only_df)
        system_no_process_df = system_no_process_df.drop(SystemColumns.ENERGY_USAGE_SYSTEM_COL, axis=1)

        # Predict energy consumption for system-process
        system_no_process_energy_predictions = self.__system_model.predict(system_no_process_df)

        # Calculate process energy consumption
        process_energy_predictions = system_only_df[
                                         SystemColumns.ENERGY_USAGE_SYSTEM_COL] - system_no_process_energy_predictions

        return process_energy_predictions

    def __build_df_without_process(self, system_only_df: pd.DataFrame, process_only_df: pd.DataFrame) -> pd.DataFrame:
        system_no_process_df = system_only_df.copy()
        for system_col, process_col in COLUMNS_MAPPING.items():
            system_no_process_df[system_col] = system_only_df[system_col] - process_only_df[process_col]

        return system_no_process_df

    def __build_df_with_target(self, full_df: pd.DataFrame, process_energy: pd.Series) -> pd.DataFrame:
        """
        This method concatenates the process energy usage (calculated in previous steps) with the original dataframe,
        filters irrelevant rows and selects the relevant features.

        (Performs part of step 6 in the pipeline)
        """
        # Build process+system dataset
        full_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = process_energy
        full_df_processed_with_energy = full_df[
            full_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL].notna()]

        full_df_processed_with_energy_filtered = self.__process_energy_filter_manager.filter_dataset(
            full_df_processed_with_energy)
        process_system_df = ProcessAndSystemFeatureSelector().select_features(full_df_processed_with_energy_filtered)
        process_system_df = process_system_df.drop(SystemColumns.ENERGY_USAGE_SYSTEM_COL, axis=1)
        process_system_df.to_csv(PROCESS_SYSTEM_DF_PATH)
        return process_system_df

    def __train_process_energy_model(self, process_system_df: pd.DataFrame):
        """
        This method trains a regression model using the processed dataframe, where the target is the process energy usage.

        (Performs step 6 in the pipeline)
        """
        # Train full energy measurement model
        process_model, process_scaler = self._run_pipeline_executor(process_system_df,
                                                                    ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                                                                    best_model_metric_name=BEST_PROCESS_MODEL_METRIC,
                                                                    hyper_parameters={
                                                                      "loss": "squared_error",
                                                                      "learning_rate": 0.06,
                                                                      "max_iter": 300,
                                                                      "max_depth": 11,
                                                                      "min_samples_leaf": 20,
                                                                      "l2_regularization": 0.05,
                                                                      "max_bins": 255,
                                                                      "early_stopping": True,
                                                                      "validation_fraction": 0.1,
                                                                      "n_iter_no_change": 20,
                                                                      "random_state": 42})

        # Save elements to future use
        self._model = process_model
        self._scaler = process_scaler
