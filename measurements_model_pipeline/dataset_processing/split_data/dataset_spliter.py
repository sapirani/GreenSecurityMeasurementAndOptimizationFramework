from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from typing import Optional

from measurements_model_pipeline.column_names import ProcessColumns


class DatasetSpliter(ABC):
    def __init__(self, train_path: str, test_path: str, full_dataset_path: str):
        self._train_path = train_path
        self._test_path = test_path
        self._full_dataset_path = full_dataset_path

    @abstractmethod
    def _train_test_split(self, x: pd.DataFrame, y: pd.Series) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

    def __save_split(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        full_train_set = pd.concat([x_train, y_train], axis=1)
        full_test_set = pd.concat([x_test, y_test], axis=1)

        full_train_set.to_csv(self._train_path)
        full_test_set.to_csv(self._test_path)

    @classmethod
    def _extract_dataframe_if_exists(cls, dataset_path: str) -> Optional[tuple[pd.DataFrame, pd.Series]]:
        dataset_file_path = Path(dataset_path)
        if dataset_file_path.exists():
            dataset = pd.read_csv(dataset_file_path, index_col=0)
            X = dataset.drop(columns=[ProcessColumns.ENERGY_USAGE_PROCESS_COL])
            y = dataset[ProcessColumns.ENERGY_USAGE_PROCESS_COL]
            return X, y

        return None

    def __extract_train_test(self) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        train_splits = self._extract_dataframe_if_exists(self._train_path)
        test_splits = self._extract_dataframe_if_exists(self._test_path)

        if train_splits is not None and test_splits is not None:
            X_train, y_train = train_splits
            X_test, y_test = test_splits
            return X_train, X_test, y_train, y_test

        return None

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        existing_splits = self.__extract_train_test()
        if existing_splits is not None:
            X_train, X_test, y_train, y_test = existing_splits
            return X_train, X_test, y_train, y_test

        else:
            full_dataset = self._extract_dataframe_if_exists(self._full_dataset_path)
            if full_dataset is None:
                raise ValueError('Dataset not found at {}'.format(self._full_dataset_path))
            X, y = full_dataset
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            self.__save_split(X_train, X_test, y_train, y_test)
            return X_train, X_test, y_train, y_test
