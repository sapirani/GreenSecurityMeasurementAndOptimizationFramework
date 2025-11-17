import os
from abc import ABC, abstractmethod
import pandas as pd

from energy_model.dataset_processing.scalers.data_scaler import DataScaler
from energy_model.models.model import Model
from energy_model.pipelines.model_pipeline_executor import PipelineExecutor

MODEL_FILE_PATH = "energy_model.pickle"
SCALER_FILE_PATH = "energy_scaler.pickle"


class AbstractEnergyModel(ABC):
    def __init__(self):
        self._model = None
        self._scaler = None

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
