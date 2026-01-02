from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class TargetCalculator(ABC):
    def __init__(self, target_column: str, must_appear_columns: list[Union[str | tuple[str]]]):
        self._target_column = target_column
        self._must_appear_columns = must_appear_columns

    def add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.__verify_must_have_columns(df):
            return self._add_target_to_dataframe(df)
        return df

    def __verify_must_have_columns(self, df: pd.DataFrame) -> bool:
        return set(self._must_appear_columns).issubset(set(df.columns))

    @abstractmethod
    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
