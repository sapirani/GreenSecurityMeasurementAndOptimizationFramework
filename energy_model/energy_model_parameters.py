import os
from pathlib import Path

# For saving the dataset (before processing)
FULL_DATASET_BEFORE_PROCESSING_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\dataset_creation\full_dataset_system_based.csv"

# For saving dataset after all processing
PROCESS_SYSTEM_DF_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\system_process_df.csv"

# For saving the energy model
DEFAULT_ENERGY_MODEL_PATH = r"energy_model_elements"
BASE_MODEL_DIR = Path(__file__).parent.parent
MODEL_FILE_NAME = os.path.join(BASE_MODEL_DIR, r"energy_prediction_model.pkl")

# For grid search
GRID_SEARCH_TEST_RESULTS_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\energy_model_elements\grid_search_results.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\best_estimators_results.csv"
RESULTS_TOP_MODELS_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\energy_model_elements\top_estimators_results.csv"  # fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\measurements_results\top_estimators_results"
