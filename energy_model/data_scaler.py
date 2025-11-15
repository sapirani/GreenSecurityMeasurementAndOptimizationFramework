import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataScaler:
    def __init__(self):
        self.__scaler = StandardScaler()  # todo: change to read from file

    def fit(self, X: pd.DataFrame):
        self.__scaler.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled_x = self.__scaler.transform(X)
        return pd.DataFrame(
            scaled_x,
            columns=X.columns,
            index=X.index
        )
