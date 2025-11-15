import joblib
import pandas as pd

from energy_model.configs.paths_config import FULL_DATASET_BEFORE_PROCESSING_PATH
from energy_model.models.energy_prediction_model import EnergyPredictionModel

if __name__ == "__main__":
    energy_model = EnergyPredictionModel()
    full_df = pd.read_csv(FULL_DATASET_BEFORE_PROCESSING_PATH, index_col=0)
    try:
        energy_model.build_energy_model(full_df)
    except Exception as e:
        print("Model was loaded from file :) You can use the model now.")
    print("Done")

    joblib.dump("energy_model.pkl", energy_model)