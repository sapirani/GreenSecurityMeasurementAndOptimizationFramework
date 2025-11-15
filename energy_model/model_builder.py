import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from energy_model.model import Model


class ModelBuilder:
    @staticmethod
    def build_model(df: pd.DataFrame, target_column: str):
        """
        The method creates an instance of the energy model.
        Input:
            df: dataframe with processed and filtered data (after scaling).
            target_column: name of the target column.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = Model()
        model.fit(X, y)
        return model