from abc import ABC, abstractmethod

import pandas as pd


class AbstractFilter(ABC):
    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
