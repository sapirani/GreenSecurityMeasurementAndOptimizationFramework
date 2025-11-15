import pandas as pd

from energy_model.dataset_processing.feature_selection.feature_selector import FeatureSelector
from energy_model.dataset_processing.filters.abstract_filter import AbstractFilter


class DataProcessor:
    def __init__(self, feature_selector: FeatureSelector, filters: list[AbstractFilter]):
        self.__feature_selector = feature_selector
        self.__filters = filters

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        return self.__feature_selector.select_features(processed_df)

    def filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df.copy()
        for df_filter in self.__filters:
            filtered_df = df_filter.filter(filtered_df)
        return filtered_df
