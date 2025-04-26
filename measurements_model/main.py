from measurements_model.config import ProcessColumns
from measurements_model.dataset_creation.dataset_creator import DatasetCreator
from measurements_model.dataset_creation.dataset_utils import save_df_to_excel
from measurements_model.dataset_processing.process_data.dataset_processor import DatasetProcessor
from measurements_model.dataset_processing.process_data.feature_selection.all_features_no_energy_selector import \
    AllFeaturesNoEnergy
from measurements_model.dataset_processing.process_data.feature_selection.only_process_with_hardware_feature_selector import \
    ProcessAndHardware
from measurements_model.dataset_processing.process_data.feature_selection.process_and_full_system_feature_selector import \
    ProcessAndTotalSystem
from measurements_model.dataset_processing.process_data.feature_selection.process_and_system_no_hardware_feature_selector import \
    ProcessAndSystemNoHardware
from measurements_model.dataset_processing.split_data.regular_spliter import RegularDatasetSplitter

IDLE_DIR_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\idle\Measurement 427"
ALL_MEASUREMENTS_DIR_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\measurements_with_resources"
FULL_DATASET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_dataset.csv"
FULL_PREPROCESSED_DATASET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_preprocessed_dataset.csv"

DF_ALL_FEATURES_NO_ENERGY_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\all_features_no_energy_preprocessed_dataset.csv"
DF_WITHOUT_SYSTEM_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\without_system_preprocessed_dataset.csv"
DF_PROCESS_AND_FULL_SYSTEM_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\process_and_full_system_preprocessed_dataset.csv"
DF_WITHOUT_HARDWARE_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\without_hardware_preprocessed_dataset.csv"

TRAIN_SET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\train_set.csv"
TEST_SET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\test_set.csv"

if __name__ == '__main__':
    dataset_creator = DatasetCreator(idle_dir_path=IDLE_DIR_PATH, measurements_dir_path=ALL_MEASUREMENTS_DIR_PATH)
    df = dataset_creator.create_dataset()
    print(df)
    save_df_to_excel(df, FULL_DATASET_PATH)
    dataset_processor = DatasetProcessor(ProcessColumns.ENERGY_USAGE_PROCESS_COL)
    preprocessed_df = dataset_processor.preprocess_dataset(df)
    save_df_to_excel(preprocessed_df, FULL_PREPROCESSED_DATASET_PATH)

    feature_selector = AllFeaturesNoEnergy()
    df_all_features_no_energy = feature_selector.select_features(preprocessed_df)
    save_df_to_excel(df_all_features_no_energy, DF_ALL_FEATURES_NO_ENERGY_PATH)

    feature_selector = ProcessAndHardware()
    df_without_system = feature_selector.select_features(preprocessed_df)
    save_df_to_excel(df_without_system, DF_WITHOUT_SYSTEM_PATH)

    feature_selector = ProcessAndTotalSystem()
    df_process_and_full_system = feature_selector.select_features(preprocessed_df)
    save_df_to_excel(df_process_and_full_system, DF_PROCESS_AND_FULL_SYSTEM_PATH)

    feature_selector = ProcessAndSystemNoHardware()
    df_without_hardware = feature_selector.select_features(df)
    save_df_to_excel(df_without_hardware, DF_WITHOUT_HARDWARE_PATH)

    dataset_splitter = RegularDatasetSplitter(train_path=TRAIN_SET_PATH, test_path=TEST_SET_PATH, full_dataset_path=FULL_DATASET_PATH, test_size=0.2)
    X_train, X_test, y_train, y_test = dataset_splitter.split_data()