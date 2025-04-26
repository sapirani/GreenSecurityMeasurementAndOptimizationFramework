from measurements_model.dataset_creation.dataset_creator import DatasetCreator
from measurements_model.dataset_creation.dataset_utils import save_df_to_excel
from measurements_model.dataset_processing.process_data.preprocess_dataset import preprocess_dataset

IDLE_DIR_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\idle\Measurement 427"
ALL_MEASUREMENTS_DIR_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\measurements_with_resources"
FULL_DATASET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_dataset.csv"
FULL_PREPROCESSED_DATASET_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\full_preprocessed_dataset.csv"
if __name__ == '__main__':
    dataset_creator = DatasetCreator(idle_dir_path=IDLE_DIR_PATH, measurements_dir_path=ALL_MEASUREMENTS_DIR_PATH)
    df = dataset_creator.create_dataset()
    print(df)
    save_df_to_excel(df, FULL_DATASET_PATH)
    # processed_df = preprocess_dataset(df)
    # save_df_to_excel(processed_df, FULL_PREPROCESSED_DATASET_PATH)