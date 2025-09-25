from abc import ABC, abstractmethod

import pandas as pd


class Processor(ABC):
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass