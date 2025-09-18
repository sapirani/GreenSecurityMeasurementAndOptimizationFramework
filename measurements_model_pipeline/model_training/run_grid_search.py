from measurements_model_pipeline.dataset_parameters import FULL_PREPROCESSED_DATASET_PATH, TRAIN_SET_PATH, TEST_SET_PATH
from measurements_model_pipeline.column_names import ProcessColumns
from measurements_model_pipeline.dataset_processing.feature_selection.process_and_system_feature_selector import \
    ProcessAndSystemFeatureSelector
from measurements_model_pipeline.model_execution.dataset_pipeline_executor import DatasetPipelineExecutor
from measurements_model_pipeline.dataset_processing.split_data.regular_spliter import RegularDatasetSplitter
from measurements_model_pipeline.model_training.models_config import MODELS_WITHOUT_PARAMETERS, REGRESSION_MODELS_WITH_PARAMETERS
from measurements_model_pipeline.model_training.search_best_model import ModelSelector

GRID_SEARCH_METRICS = ["neg_mean_absolute_error"]


def run_grid_search():
    feature_selector = ProcessAndSystemFeatureSelector()
    dataset_splitter = RegularDatasetSplitter(TRAIN_SET_PATH, TEST_SET_PATH, FULL_PREPROCESSED_DATASET_PATH)
    dataset_pipeline = DatasetPipelineExecutor(energy_column_to_filter_by=ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                                               feature_selector=feature_selector, dataset_spliter=dataset_splitter)
    X_train, X_test, y_train, y_test = dataset_pipeline.split_dataset()
    model_selector = ModelSelector(models_to_experiment=REGRESSION_MODELS_WITH_PARAMETERS)
    best_model = model_selector.choose_best_model(GRID_SEARCH_METRICS, X_train, X_test, y_train, y_test)
    print("Setup:")
    print("Features: Process and System only (no network, idle or hardware). No energy of system.")
    print("5 splits and 5 best models.")
    print(f"\nThe best model is:\n{best_model}")


if __name__ == "__main__":
    run_grid_search()
