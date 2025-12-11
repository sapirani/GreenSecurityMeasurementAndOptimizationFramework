import logging

import pandas as pd

from energy_model.dataset_processing.filters.negative_value_filter import NegativeValueFilter


class EnergyFilter(NegativeValueFilter):
    def __init__(self, energy_column: str):
        super().__init__([energy_column])
        self.__energy_column = energy_column

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        negative_mask = df[self.__energy_column] < 0
        negative_indices = df.index[negative_mask]

        if len(negative_indices) > 0:
            logging.warning(
                f"Negative values found in energy column '{self.__energy_column}' "
                f"at indices: {list(negative_indices)}"
            )

        # Apply the parent's filtering logic
        filtered_df = super().filter(df)

        return filtered_df
