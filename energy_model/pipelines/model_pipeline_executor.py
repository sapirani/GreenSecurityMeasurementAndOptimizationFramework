import pandas as pd

from energy_model.dataset_processing.scalers.data_scaler import DataScaler
from energy_model.evaluation.model_evaluator import ModelEvaluator
from energy_model.models.model import Model
from energy_model.pipelines.pipeline_utils import split_train_test


class ModelPipelineExecutor:
    def __init__(self, target_column: str):
        self.__target_column = target_column
        self.__model_evaluator = ModelEvaluator()

    def build_train_test(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return split_train_test(df, self.__target_column)

    def build_scaler(self, X: pd.DataFrame) -> DataScaler:
        scaler = DataScaler()
        scaler.fit(X)
        return scaler

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series, scaler: DataScaler) -> Model:
        X_train_scaled = scaler.transform(X_train)
        model = Model()
        model.fit(X_train_scaled, y_train)
        return model

    def evaluate_model(self, model: Model, X_test: pd.DataFrame, y_test: pd.Series, scaler: DataScaler):
        X_test_scaled = scaler.transform(X_test)
        y_pred = pd.Series(model.predict(X_test_scaled)).reset_index(drop=True)

        negative_predictions_mask = y_pred.lt(0)
        if negative_predictions_mask.any():
            negative_predictions = y_pred[negative_predictions_mask]
            print("Negative predictions found:")
            print(f"  Values : {negative_predictions.tolist()}")
            print(f"  Indices: {negative_predictions.index.tolist()}")

        results = self.__model_evaluator.evaluate(y_test, y_pred)
        self.__model_evaluator.print_results(results)
