import os
from pathlib import Path

SCORING_METHODS_FOR_MODEL = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

BASE_MODEL_DIR = Path(__file__).parent
MODEL_FILE_NAME = os.path.join(BASE_MODEL_DIR, r"energy_prediction_model.pkl")
