import pandas as pd

from energy_model.dataset_processing.filters.abstract_filter import AbstractFilter


class NegativeValueFilter(AbstractFilter):
    def __init__(self, columns_to_filter_by: list[str]):
        self._columns_to_filter_by = columns_to_filter_by

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self._columns_to_filter_by:
            df = df[df[column] >= 0]
        return df
