import pandas as pd
from sklearn.model_selection import train_test_split

from measurements_model.dataset_processing.split_data.dataset_spliter import DatasetSpliter


class RegularDatasetSplitter(DatasetSpliter):
    def __init__(self, train_path: str, test_path: str, full_dataset_path: str, test_size: float = 0.2):
        super().__init__(train_path, test_path, full_dataset_path)
        self.test_size = test_size

    def _train_test_split(self, x: pd.DataFrame, y: pd.Series) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(x, y, test_size=self.test_size)