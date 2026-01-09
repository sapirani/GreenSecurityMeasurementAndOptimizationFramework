from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class BestModelConfig:
    MODEL_NAME = HistGradientBoostingRegressor
    MODEL_HYPER_PARAMETERS = {
        "max_depth": 11,
        "min_samples_leaf": 10
    }


class Model:
    def __init__(self, best_model_hyper_parameters: dict[str, Any] = None):
        if best_model_hyper_parameters is None:
            best_model_hyper_parameters = BestModelConfig.MODEL_HYPER_PARAMETERS
        self.__best_model_hyper_parameters = best_model_hyper_parameters
        self.__model = self.__initialize_model()

    def __initialize_model(self):
        print(f"Model's configuration: \n{self.__best_model_hyper_parameters}\n\n")
        return BestModelConfig.MODEL_NAME(**self.__best_model_hyper_parameters)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.__model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.__model.predict(X)
