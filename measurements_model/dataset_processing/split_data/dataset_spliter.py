from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from typing import Optional


class DatasetSpliter(ABC):
    def __init__(self, train_path: str, test_path: str, full_dataset_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.full_dataset_path = full_dataset_path

    @abstractmethod
    def _train_test_split(self, x: pd.DataFrame, y: pd.Series) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

    def __save_split(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        full_train_set = pd.concat([x_train, y_train], axis=1)
        full_test_set = pd.concat([x_test, y_test], axis=1)

        full_train_set.to_csv(self.train_path)
        full_test_set.to_csv(self.test_path)

    @classmethod
    def __extract_dataframe_if_exists(cls, dataset_path: str) -> Optional[tuple[pd.DataFrame, pd.Series]]:
        dataset_file_path = Path(dataset_path)
        if dataset_file_path.exists():
            dataset = pd.read_csv(dataset_file_path)
            X = dataset.drop(columns=['label'])
            y = dataset['label']
            return X, y

        return None

    def __extract_train_test(self) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        train_splits = self.__extract_dataframe_if_exists(self.train_path)
        test_splits = self.__extract_dataframe_if_exists(self.test_path)

        if train_splits is not None and test_splits is not None:
            X_train, y_train = train_splits
            X_test, y_test = test_splits
            return X_train, X_test, y_train, y_test

        return None

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        splits = self.__extract_train_test()
        if splits is not None:
            X_train, X_test, y_train, y_test = splits
            return X_train, X_test, y_train, y_test

        else:
            X, y = self.__extract_dataframe_if_exists(self.full_dataset_path)
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            self.__save_split(X_train, X_test, y_train, y_test)
            return X_train, X_test, y_train, y_test
