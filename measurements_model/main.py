from measurements_model.dataset_creation.dataset_creator import DatasetCreator
from measurements_model.dataset_creation.dataset_utils import save_df_to_excel

IDLE_DIR_PATH = fr""
ALL_MEASUREMENTS_DIR_PATH = fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results"
DATASET_PATH = fr""

if __name__ == '__main__':
    dataset_creator = DatasetCreator(idle_dir_path=IDLE_DIR_PATH, measurements_dir_path=ALL_MEASUREMENTS_DIR_PATH)
    df = dataset_creator.create_dataset()
    print(df)
    save_df_to_excel(df, DATASET_PATH)