import logging

import pandas as pd


class EnergyFilter:
    def __init__(self, energy_threshold: float, energy_column: str):
        self.__energy_threshold = energy_threshold
        self.__energy_column = energy_column

    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = data[self.__energy_column] >= self.__energy_threshold

        # rows that do NOT meet the condition
        bad_rows = data[~mask]

        if not bad_rows.empty:
            logging.warning(
                "Some values for process energy turned out negative after calculating total - idle energy. "
                f"Filtered out rows with indices: {bad_rows.index.tolist()}"
            )

        return data[mask]
