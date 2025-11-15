import os
from pathlib import Path

FULL_DATASET_BEFORE_PROCESSING_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\dataset_creation\full_dataset_system_based.csv"

DEFAULT_ENERGY_MODEL_PATH = r"energy_model_elements"

BASE_MODEL_DIR = Path(__file__).parent.parent
MODEL_FILE_NAME = os.path.join(BASE_MODEL_DIR, r"energy_prediction_model.pkl")
