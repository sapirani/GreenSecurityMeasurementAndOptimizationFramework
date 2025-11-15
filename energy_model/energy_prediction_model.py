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

        system_model, system_scaler = self.__build_and_evaluate_system_model(full_df)
        system_only_df = self.__system_only_feature_selector.select_features(full_df)
        process_only_df = ProcessOnlyFeatureSelector().select_features(full_df)

        system_no_process_df = self.__build_df_without_process(system_only_df, process_only_df)

        system_no_process_scaled_df = system_scaler.transform(system_no_process_df)
        system_no_process_energy_predictions = system_model.predict(system_no_process_scaled_df)

        process_energy_predictions = system_only_df[
                                         SystemColumns.ENERGY_USAGE_SYSTEM_COL] - system_no_process_energy_predictions

        full_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = process_energy_predictions
        full_df = full_df[full_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL].notna()]

        process_model, process_scaler = self.__build_and_evaluate_process_model(full_df)
        self.__model = process_model
        self.__scaler = process_scaler

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.__model is None:
            raise Exception('You should call build_energy_model first!')

        df_scaled = self.__scaler.transform(df)
        return self.__model.predict(df_scaled)

    def __build_and_evaluate_system_model(self, full_df: pd.DataFrame) -> tuple[Model, DataScaler]:
        return self.__run_pipeline_executor(full_df, self.__system_only_feature_selector,
                                            SystemColumns.ENERGY_USAGE_SYSTEM_COL)

    def __build_and_evaluate_process_model(self, full_df: pd.DataFrame) -> tuple[Model, DataScaler]:
        return self.__run_pipeline_executor(full_df, ProcessAndSystemFeatureSelector(),
                                            ProcessColumns.ENERGY_USAGE_PROCESS_COL)

    def __build_df_without_process(self, system_only_df: pd.DataFrame, process_only_df: pd.DataFrame) -> pd.DataFrame:
        system_no_process_df = system_only_df.copy()
        for system_col, process_col in COLUMNS_MAPPING.items():
            system_no_process_df[system_col] = system_only_df[system_col] - process_only_df[process_col]

        return system_no_process_df

    def __run_pipeline_executor(self, full_df: pd.DataFrame, feature_selector: FeatureSelector,
                                target_col: str) -> tuple[Model, DataScaler]:
        data_processor = DataProcessor(
            feature_selector=feature_selector,
            filters=DEFAULT_FILTERS
        )

        model_pipeline = PipelineExecutor(data_processor, target_col)
        X_train, X_test, y_train, y_test = model_pipeline.build_train_test(full_df)
        scalar = model_pipeline.build_scaler(X_train)
        model = model_pipeline.build_model(X_train, y_train, scalar)
        model_pipeline.evaluate_model(model, X_test, y_test, scalar)
        return model, scalar
