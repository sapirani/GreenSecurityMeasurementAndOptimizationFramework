import pandas as pd

from energy_model.dataset_processing.filters.abstract_filter import AbstractFilter
from energy_model.energy_model_parameters import DEFAULT_MAX_QUANTILE_VALUE, DEFAULT_MIN_QUANTILE_VALUE


class OutlierFilter(AbstractFilter):
    def __init__(self, outliers_columns: list[str], min_quantile_value: float = DEFAULT_MIN_QUANTILE_VALUE,
                 max_quantile_value: float = DEFAULT_MAX_QUANTILE_VALUE):
        self.__outliers_columns = outliers_columns
        self.__min_quantile_value = min_quantile_value
        self.__max_quantile_value = max_quantile_value

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.__outliers_columns:
            low = df[col].quantile(self.__min_quantile_value)
            high = df[col].quantile(self.__max_quantile_value)
            df = df[(df[col] >= low) & (df[col] <= high)]
        return df
