import pandas as pd

from energy_model.configs.columns import SystemColumns, ProcessColumns, COLUMNS_MAPPING
from energy_model.configs.defaults_configs import DEFAULT_FILTERS
from energy_model.data_scaler import DataScaler
from energy_model.dataset_processing.data_processor import DataProcessor
from energy_model.dataset_processing.feature_selection.feature_selector import FeatureSelector
from energy_model.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector
from energy_model.dataset_processing.feature_selection.process_only_feature_selector import ProcessOnlyFeatureSelector
from energy_model.dataset_processing.feature_selection.system_only_feature_selector import SystemOnlyFeatureSelector
from energy_model.dataset_processing.filters.energy_filter import EnergyFilter
from energy_model.model import Model
from energy_model.pipelines.pipeline_executor import PipelineExecutor


class EnergyPredictionModel:
    def __init__(self):
        self.__model = None  # todo: initialize from file if exists
        self.__scaler = None
        self.__system_only_feature_selector = SystemOnlyFeatureSelector()

    def build_energy_model(self, full_df: pd.DataFrame):
        if self.__model is not None:
            raise Exception('This method should be called once!')

        system_data_processor = DataProcessor(
            feature_selector=SystemOnlyFeatureSelector(),
            filters=DEFAULT_FILTERS
        )

        # filter irrelevant rows
        full_df_processed = system_data_processor.filter_dataset(full_df)

        # Train a model on system columns only
        system_only_df = system_data_processor.select_features(full_df_processed)
        system_model, system_scaler = self.__build_and_evaluate_system_model(system_only_df)

        # Build System - Process dataset
        process_only_df = ProcessOnlyFeatureSelector().select_features(full_df_processed)
        system_no_process_df = self.__build_df_without_process(system_only_df, process_only_df)
        system_no_process_df = system_no_process_df.drop(SystemColumns.ENERGY_USAGE_SYSTEM_COL, axis=1)
        system_no_process_scaled_df = system_scaler.transform(system_no_process_df)

        # Predict energy consumption for system-process
        system_no_process_energy_predictions = system_model.predict(system_no_process_scaled_df)

        # Calculate process energy consumption
        process_energy_predictions = system_only_df[
                                         SystemColumns.ENERGY_USAGE_SYSTEM_COL] - system_no_process_energy_predictions


        # Build process+system dataset
        full_df_processed[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = process_energy_predictions
        full_df_processed_with_energy = full_df_processed[full_df_processed[ProcessColumns.ENERGY_USAGE_PROCESS_COL].notna()]

        process_data_processor = DataProcessor(
            feature_selector=ProcessAndSystemFeatureSelector(),
            filters=[EnergyFilter(ProcessColumns.ENERGY_USAGE_PROCESS_COL)],
        )
        full_df_processed_with_energy_filtered = process_data_processor.filter_dataset(full_df_processed_with_energy)
        process_system_df = process_data_processor.select_features(full_df_processed_with_energy_filtered)

        # Train full energy measurement model
        process_model, process_scaler = self.__build_and_evaluate_process_model(process_system_df)

        # Save elements to use
        self.__model = process_model
        self.__scaler = process_scaler

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.__model is None:
            raise Exception('You should call build_energy_model first!')

        df_scaled = self.__scaler.transform(df)
        return self.__model.predict(df_scaled)

    def __build_and_evaluate_system_model(self, full_df: pd.DataFrame) -> tuple[Model, DataScaler]:
        return self.__run_pipeline_executor(full_df, SystemColumns.ENERGY_USAGE_SYSTEM_COL)

    def __build_and_evaluate_process_model(self, full_df: pd.DataFrame) -> tuple[Model, DataScaler]:
        return self.__run_pipeline_executor(full_df, ProcessColumns.ENERGY_USAGE_PROCESS_COL)

    def __build_df_without_process(self, system_only_df: pd.DataFrame, process_only_df: pd.DataFrame) -> pd.DataFrame:
        system_no_process_df = system_only_df.copy()
        for system_col, process_col in COLUMNS_MAPPING.items():
            system_no_process_df[system_col] = system_only_df[system_col] - process_only_df[process_col]

        return system_no_process_df

    def __run_pipeline_executor(self, full_df: pd.DataFrame, target_col: str) -> tuple[Model, DataScaler]:

        model_pipeline = PipelineExecutor(target_col)
        X_train, X_test, y_train, y_test = model_pipeline.build_train_test(full_df)
        scalar = model_pipeline.build_scaler(X_train)
        model = model_pipeline.build_model(X_train, y_train, scalar)
        model_pipeline.evaluate_model(model, X_test, y_test, scalar)
        return model, scalar
