from abc import ABC
from sklearn.model_selection import train_test_split
import pandas as pd

from energy_model.data_scaler import DataScaler
from energy_model.evaluation.model_evaluator import ModelEvaluator
from energy_model.model import Model


class PipelineExecutor(ABC):
    def __init__(self, target_column: str):
        self.__target_column = target_column
        self.__model_evaluator = ModelEvaluator()

    def build_train_test(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.__target_column])
        y = df[self.__target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.copy()
        X_test = X_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()

        y_train = pd.Series(y_train.squeeze()).reset_index(drop=True)
        y_test = pd.Series(y_test.squeeze()).reset_index(drop=True)

        return X_train, X_test, y_train, y_test

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
        y_negative_pred = [y for y in y_pred if y < 0]
        if y_negative_pred:
            print(f"there are negative predictions: {y_negative_pred}")

        results = self.__model_evaluator.evaluate(y_test, y_pred)
        self.__model_evaluator.print_results(results)
