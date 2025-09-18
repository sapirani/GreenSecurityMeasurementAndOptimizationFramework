from abc import ABC, abstractmethod

import pandas as pd


class Filter(ABC):
    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        pass