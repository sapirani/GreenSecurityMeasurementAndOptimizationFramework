import pandas as pd


class EnergyFilter:
    def __init__(self, energy_threshold: float, energy_column: str):
        self.__energy_threshold = energy_threshold
        self.__energy_column = energy_column

    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data[self.__energy_column] >= self.__energy_threshold]
        return data