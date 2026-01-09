from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from energy_model.configs.defaults_configs import DEFAULT_CV_SPLITS_N, DEFAULT_BEST_MODEL_METRIC, DEFAULT_FILTERS
from energy_model.dataset_processing.filters.filter_manager import FilterManager
from energy_model.dataset_processing.scalers.data_scaler import DataScaler
from energy_model.models.model import Model
from energy_model.pipelines.model_pipeline_executor import ModelPipelineExecutor

MODEL_FILE_PATH = "energy_model.pickle"
SCALER_FILE_PATH = "energy_scaler.pickle"
TARGET_COLUMN = "target"

class AbstractEnergyModel(ABC):
    def __init__(self):
        self._model = None
        self._scaler = None
        self._full_df_filter_manager = FilterManager(filters=DEFAULT_FILTERS)

    @abstractmethod
    def build_energy_model(self, full_df: pd.DataFrame):
        pass

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise Exception('You should call build_energy_model first!')

        df_scaled = self._scaler.transform(df)
        return self._model.predict(df_scaled)

    def _run_pipeline_executor(self, full_df: pd.DataFrame, target_col: str, n_splits: int = DEFAULT_CV_SPLITS_N,
                               best_model_metric_name: str = DEFAULT_BEST_MODEL_METRIC,
                               hyper_parameters: dict[str, Any] = None) -> tuple[Model, DataScaler]:
        model_pipeline = ModelPipelineExecutor(target_col)
        cv_splits = model_pipeline.build_train_test_cv(full_df, n_splits=n_splits)

        best_model = None
        best_scaler = None
        best_score = None
        best_dataset_split = None

        for fold_id, (X_train, X_test, y_train, y_test) in enumerate(cv_splits, start=1):
            print(f"\n--- Running Fold {fold_id}/{n_splits} ---")

            scaler = model_pipeline.build_scaler(X_train)
            model = model_pipeline.build_model(X_train, y_train, scaler, hyper_parameters)
            results = model_pipeline.evaluate_model(model, X_test, y_test, scaler)

            score = results.get(best_model_metric_name)
            if score is None:
                raise ValueError(
                    f"Metric '{best_model_metric_name}' not found in evaluation result: {results}"
                )

            # Update best model
            if best_score is None or score < best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_dataset_split = (X_train, X_test, y_train, y_test)
                print(f"New best model found on fold {fold_id} with {best_model_metric_name}={score}")

        self.__save_best_split(best_dataset_split, target_col)
        return best_model, best_scaler

    def __save_best_split(self, best_dataset_split: tuple, target_col: str):
        best_x_train, best_x_test, best_y_train, best_y_test = best_dataset_split
        best_train_set = best_x_train
        best_train_set[TARGET_COLUMN] = best_y_train
        best_test_set = best_x_test
        best_test_set[TARGET_COLUMN] = best_y_test
        best_train_set.to_csv(f"train_{target_col}.csv", index=False)
        best_test_set.to_csv(f"test_{target_col}.csv", index=False)