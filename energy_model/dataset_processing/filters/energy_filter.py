import logging

import pandas as pd

from energy_model.dataset_processing.filters.negative_value_filter import NegativeValueFilter


class EnergyFilter(NegativeValueFilter):
    def __init__(self, energy_column: str):
        super().__init__([energy_column])

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = super().filter(df)

        if filtered_df.shape[0] != df.shape[0]:
            logging.warning(f"Some values for energy in column {self._columns_to_filter_by} turned out negative.")

        return filtered_df