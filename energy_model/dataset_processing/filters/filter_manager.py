import pandas as pd

from energy_model.dataset_processing.filters.abstract_filter import AbstractFilter


class FilterManager:
    def __init__(self, filters: list[AbstractFilter]):
        self.__filters = filters

    def filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df.copy()
        for df_filter in self.__filters:
            filtered_df = df_filter.filter(filtered_df)
        return filtered_df