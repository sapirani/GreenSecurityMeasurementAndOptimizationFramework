import pandas as pd

from measurements_model.config import ProcessColumns
from measurements_model.dataset_processing.process_data.filters.energy_filter import EnergyFilter
from measurements_model.dataset_processing.process_data.processors.categorial_variable_processor import \
    CategoricalVariableProcessor


class DatasetProcessor:
    def __init__(self, energy_column: str):
        self.__filters = [EnergyFilter(energy_threshold=-1, energy_column=energy_column)]
        self.__processors = [CategoricalVariableProcessor()]

    def __filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        for filter_for_df in self.__filters:
            df = filter_for_df.filter_data(df)

        return df

    def __process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        for processor_for_df in self.__processors:
            df = processor_for_df.process_data(df)

        return df

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = self.__filter_dataset(df)
        processed_df = self.__process_dataset(filtered_df)
        # TODO: can add processing, e.g. normalizing columns, etc.
        return processed_df
