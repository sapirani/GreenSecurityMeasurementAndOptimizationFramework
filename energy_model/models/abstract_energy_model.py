import os
from abc import ABC, abstractmethod

import joblib
import pandas as pd

from energy_model.dataset_processing.scalers.data_scaler import DataScaler
from energy_model.energy_model_parameters import DEFAULT_ENERGY_MODEL_PATH
from energy_model.models.model import Model
from energy_model.pipelines.model_pipeline_executor import PipelineExecutor

MODEL_FILE_PATH = "energy_model.pickle"
SCALER_FILE_PATH = "energy_scaler.pickle"


class AbstractEnergyModel(ABC):
    def __init__(self, saved_info_dir_path: str = None):
        self._model = None
        self._scaler = None
        self._results_dir_path = saved_info_dir_path if saved_info_dir_path else DEFAULT_ENERGY_MODEL_PATH
        self.__initialize_model_and_scaler(self._results_dir_path)

    def __initialize_model_and_scaler(self, dir_path: str):
        if os.path.exists(os.path.join(dir_path, MODEL_FILE_PATH)) and os.path.exists(os.path.join(dir_path, SCALER_FILE_PATH)):
            self._model = joblib.load(os.path.join(dir_path, MODEL_FILE_PATH))
            self._scaler = joblib.load(os.path.join(dir_path, SCALER_FILE_PATH))

    def _save_model_and_scaler(self, dir_path: str, model_name: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(self._scaler, os.path.join(dir_path, f"{model_name}_{SCALER_FILE_PATH}"))
        joblib.dump(self._model, os.path.join(dir_path, f"{model_name}_{MODEL_FILE_PATH}"))

    @abstractmethod
    def build_energy_model(self, full_df: pd.DataFrame):
        pass

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise Exception('You should call build_energy_model first!')

        df_scaled = self._scaler.transform(df)
        return self._model.predict(df_scaled)

    def _run_pipeline_executor(self, full_df: pd.DataFrame, target_col: str) -> tuple[Model, DataScaler]:
        model_pipeline = PipelineExecutor(target_col)
        X_train, X_test, y_train, y_test = model_pipeline.build_train_test(full_df)
        scalar = model_pipeline.build_scaler(X_train)
        model = model_pipeline.build_model(X_train, y_train, scalar)
        model_pipeline.evaluate_model(model, X_test, y_test, scalar)
        return model, scalar
