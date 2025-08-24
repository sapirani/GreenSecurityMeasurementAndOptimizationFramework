import joblib

from measurements_model.config import TRAIN_SET_PATH, FULL_PREPROCESSED_DATASET_PATH, IDLE_DIR_PATH, TEST_SET_PATH, \
    ProcessColumns
from measurements_model.model_execution.dataset_pipeline_executor import DatasetPipelineExecutor
from measurements_model.dataset_processing.feature_selection.process_and_system_no_hardware_feature_selector import \
    ProcessAndSystemNoHardware
from measurements_model.dataset_processing.split_data.regular_spliter import RegularDatasetSplitter
from measurements_model.model_execution.main_model import BestModelConfig
from measurements_model.model_execution.main_model_configuration import ALL_MEASUREMENTS_DIRS_PATH, MODEL_FILE_NAME

if __name__ == '__main__':
    feature_selector = ProcessAndSystemNoHardware()
    dataset_splitter = RegularDatasetSplitter(TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH)
    dataset_pipeline = DatasetPipelineExecutor(idle_measurement_path=IDLE_DIR_PATH,
                                               all_measurement_path=ALL_MEASUREMENTS_DIRS_PATH,
                                               energy_column_to_filter_by=ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                                               feature_selector=feature_selector, dataset_spliter=dataset_splitter)
    X_train, X_test, y_train, y_test = dataset_pipeline.split_dataset()

    model = BestModelConfig.MODEL_NAME(**BestModelConfig.MODEL_PARAMETERS)  # todo: change accordingly
    model.fit(X_train, y_train)
    # Save the model as a pickle in a file
    joblib.dump(model, MODEL_FILE_NAME)

    # Load the model from the file
    knn_from_joblib = joblib.load(MODEL_FILE_NAME)

    # Use the loaded model to make predictions
    knn_from_joblib.predict(X_test)
