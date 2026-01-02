from abc import ABC, abstractmethod

import pandas as pd


class TargetCalculator(ABC):
    def __init__(self, target_column: str, must_appear_columns: list[str]):
        self._target_column = target_column
        self._must_appear_columns = must_appear_columns

    @abstractmethod
    def add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _verify_must_have_columns(self, df: pd.DataFrame) -> bool:
        return set(self._must_appear_columns).issubset(set(df.columns))
