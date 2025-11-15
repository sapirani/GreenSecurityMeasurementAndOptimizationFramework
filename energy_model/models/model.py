import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class Model:
    def __init__(self):
        self.__model = self.__initialize_model()

    def __initialize_model(self):
        # Todo: change model to read from configuration if exists
        return HistGradientBoostingRegressor(max_depth=11, min_samples_leaf=10)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.__model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.__model.predict(X)
