from pathlib import Path

import pandas as pd

from measurements_model.config import FULL_DATASET_PATH, DATASET_AFTER_FEATURE_SELECTION_PATH, \
    FULL_PREPROCESSED_DATASET_PATH
from measurements_model.dataset_creation.dataset_creator import DatasetCreator
from measurements_model.dataset_processing.feature_selection.feature_selector import FeatureSelector
from measurements_model.dataset_processing.process_data.dataset_processor import DatasetProcessor
from measurements_model.dataset_processing.split_data.dataset_spliter import DatasetSpliter


class DatasetPipelineExecutor:
    def __init__(self, idle_measurement_path: str, all_measurement_path: str, energy_column_to_filter_by: str,
                 feature_selector: FeatureSelector, dataset_spliter: DatasetSpliter):
        self.__dataset_creator = DatasetCreator(idle_dir_path=idle_measurement_path, measurements_dir_path=all_measurement_path)
        self.__dataset_processor = DatasetProcessor(energy_column_to_filter_by)
        self.__feature_selector = feature_selector
        self.__dataset_spliter = dataset_spliter

    def create_dataset(self) -> pd.DataFrame:
        full_dataset_path = Path(FULL_DATASET_PATH)
        if full_dataset_path.exists():
            return pd.read_csv(full_dataset_path, index_col=0)

        full_dataset = self.__dataset_creator.create_dataset()
        full_dataset.to_csv(FULL_DATASET_PATH)
        return full_dataset

    def process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        processed_dataset_with_selected_features_path = Path(DATASET_AFTER_FEATURE_SELECTION_PATH)
        if processed_dataset_with_selected_features_path.exists():
            return pd.read_csv(processed_dataset_with_selected_features_path, index_col=0)

        full_processed_dataset_path = Path(FULL_PREPROCESSED_DATASET_PATH)
        if full_processed_dataset_path.exists():
            full_processed_dataset = pd.read_csv(full_processed_dataset_path, index_col=0)
        else:
            full_processed_dataset = self.__dataset_processor.preprocess_dataset(dataset)
            full_processed_dataset.to_csv(FULL_PREPROCESSED_DATASET_PATH)

        dataset_with_selected_features = self.__feature_selector.select_features(full_processed_dataset)
        dataset_with_selected_features.to_csv(DATASET_AFTER_FEATURE_SELECTION_PATH)
        return dataset_with_selected_features

    def split_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self.__dataset_spliter.split_data()