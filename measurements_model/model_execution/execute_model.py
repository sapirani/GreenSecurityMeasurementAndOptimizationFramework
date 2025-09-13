import joblib
import pandas as pd

from measurements_model.config import TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH, \
    ProcessColumns, MODEL_FILE_NAME
from measurements_model.model_execution.dataset_pipeline_executor import DatasetPipelineExecutor
from measurements_model.dataset_processing.feature_selection.process_and_system_no_hardware_feature_selector import \
    ProcessAndSystemNoHardware
from measurements_model.dataset_processing.split_data.regular_spliter import RegularDatasetSplitter
from measurements_model.model_execution.measurements_model import MeasurementsModel
from measurements_model.model_training.utils import calculate_and_print_scores


def run_model():
    feature_selector = ProcessAndSystemNoHardware()
    dataset_splitter = RegularDatasetSplitter(TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH)
    dataset_pipeline = DatasetPipelineExecutor(energy_column_to_filter_by=ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                                               feature_selector=feature_selector, dataset_spliter=dataset_splitter)

    full_dataset = dataset_pipeline.create_dataset()
    processed_dataset = dataset_pipeline.process_dataset(full_dataset)

    X_train, X_test, y_train, y_test = dataset_pipeline.split_dataset()
    model = MeasurementsModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    calculate_and_print_scores(y_test, y_pred)

    joblib.dump(model, MODEL_FILE_NAME)


if __name__ == '__main__':
    run_model()
