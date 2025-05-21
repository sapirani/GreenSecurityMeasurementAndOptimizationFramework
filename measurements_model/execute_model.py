import pandas as pd

from measurements_model.config import TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH, IDLE_DIR_PATH, \
    ProcessColumns
from measurements_model.dataset_pipeline_executor import DatasetPipelineExecutor
from measurements_model.dataset_processing.feature_selection.all_features_no_energy_selector import AllFeaturesNoEnergy
from measurements_model.dataset_processing.feature_selection.process_and_system_no_hardware_feature_selector import \
    ProcessAndSystemNoHardware
from measurements_model.dataset_processing.split_data.regular_spliter import RegularDatasetSplitter
from measurements_model.main_model import MeasurementsModel
from measurements_model.main_model_configuration import ALL_MEASUREMENTS_DIRS_PATH, NETWORK_SENT_BYTES_COLUMN_NAME, \
    NETWORK_RECEIVED_BYTES_COLUMN_NAME
from measurements_model.model_training.utils import calculate_and_print_scores


def run_model():
    feature_selector = ProcessAndSystemNoHardware()
    dataset_splitter = RegularDatasetSplitter(TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH)
    dataset_pipeline = DatasetPipelineExecutor(idle_measurement_path=IDLE_DIR_PATH,
                                               all_measurement_path=ALL_MEASUREMENTS_DIRS_PATH,
                                               energy_column_to_filter_by=ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                                               feature_selector=feature_selector, dataset_spliter=dataset_splitter)

    full_dataset = dataset_pipeline.create_dataset()
    processed_dataset = dataset_pipeline.process_dataset(full_dataset)

    X_train, X_test, y_train, y_test = dataset_pipeline.split_dataset()
    model = MeasurementsModel(network_sent_bytes_column=NETWORK_SENT_BYTES_COLUMN_NAME,
                              network_received_bytes_column=NETWORK_RECEIVED_BYTES_COLUMN_NAME)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    calculate_and_print_scores(y_test, y_pred)


if __name__ == '__main__':
    run_model()
