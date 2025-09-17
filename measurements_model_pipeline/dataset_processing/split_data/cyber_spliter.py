import pandas as pd

from measurements_model_pipeline.dataset_processing.split_data.dataset_spliter import DatasetSpliter


class CyberDatasetSplitter(DatasetSpliter):
    def __init__(self, train_path: str, test_path: str, measurements_dataset_path: str, cyber_dataset_path: str):
        super().__init__(train_path=train_path, test_path=test_path, full_dataset_path=measurements_dataset_path)
        self.cyber_dataset_path = cyber_dataset_path

    def _train_test_split(self, x: pd.DataFrame, y: pd.Series) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        cyber_dataset_splits = self._extract_dataframe_if_exists(self.cyber_dataset_path)
        if cyber_dataset_splits is None:
            raise FileNotFoundError("Cyber dataset not found.")

        X_test, y_test = cyber_dataset_splits
        return x, X_test, y, y_test