import joblib
import pandas as pd

from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_PATH, PROCESS_ENERGY_MODEL_FILE_NAME, \
    SYSTEM_ENERGY_MODEL_FILE_NAME
from energy_model.models.process_energy_model import ProcessEnergyModel
from energy_model.models.system_energy_model import SystemEnergyModel

if __name__ == "__main__":
    full_df = pd.read_csv(FULL_DATASET_BEFORE_PROCESSING_PATH, index_col=0)
    system_model = SystemEnergyModel()
    system_model.build_energy_model(full_df)

    process_energy_model = ProcessEnergyModel(system_model)
    try:
        process_energy_model.build_energy_model(full_df)
    except Exception as e:
        print("Model was loaded from file :) You can use the model now.")
    print("Done")

    joblib.dump(process_energy_model, PROCESS_ENERGY_MODEL_FILE_NAME)
    joblib.dump(system_model, SYSTEM_ENERGY_MODEL_FILE_NAME)
