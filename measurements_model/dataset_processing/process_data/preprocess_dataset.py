import pandas as pd

from measurements_model.config import ProcessColumns
from measurements_model.dataset_processing.process_data.filters.energy_filter import EnergyFilter


class DatasetProcessor:
    def __init__(self, energy_column: str) -> None:
        self.__filters = [EnergyFilter(energy_threshold=0, energy_column=energy_column)]

    def __filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        for filter_for_df in self.__filters:
            df = filter_for_df.filter_data(df)

        return df

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = self.__filter_dataset(df)
        # TODO: can add processing, e.g. normalizing columns, etc.
        return filtered_df
