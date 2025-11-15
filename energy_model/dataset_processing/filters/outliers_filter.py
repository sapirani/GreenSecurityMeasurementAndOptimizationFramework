import pandas as pd

from energy_model.dataset_processing.filters.abstract_filter import AbstractFilter


class OutlierFilter(AbstractFilter):
    def __init__(self, outliers_columns: list[str]):
        self.__outliers_columns = outliers_columns

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.__outliers_columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df = df[(df[col] >= low) & (df[col] <= high)]
        return df
