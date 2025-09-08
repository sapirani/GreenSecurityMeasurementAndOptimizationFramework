import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class BestModelConfig:
    MODEL_NAME = HistGradientBoostingRegressor
    MODEL_PARAMETERS = {
        "max_depth": 11,
        "min_samples_leaf": 10
    }


class MeasurementsModel:
    def __init__(self):
        self.__model = BestModelConfig.MODEL_NAME(**BestModelConfig.MODEL_PARAMETERS)

    def fit(self, X, y):
        y = np.array(y)
        self.__model.fit(X, y)

    def predict(self, X):
        y_pred = self.__model.predict(X)
        y_pred = np.array(y_pred)
        return y_pred
