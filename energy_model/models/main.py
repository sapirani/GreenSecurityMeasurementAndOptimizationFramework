import pandas as pd

from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_PATH, PROCESS_ENERGY_MODEL_FILE_NAME, \
    SYSTEM_ENERGY_MODEL_FILE_NAME
from energy_model.models.persistence_manager import PersistenceManager
from energy_model.models.process_energy_model import ProcessEnergyModel
from energy_model.models.system_energy_model import SystemEnergyModel

if __name__ == "__main__":
    full_df = pd.read_csv(FULL_DATASET_BEFORE_PROCESSING_PATH, index_col=0)

    print("~~~~~~~~~~~~ Starting building System Energy Model ~~~~~~~~~~~~")
    if PersistenceManager.model_exists(SYSTEM_ENERGY_MODEL_FILE_NAME):
        system_model = PersistenceManager.load_model(SYSTEM_ENERGY_MODEL_FILE_NAME)
    else:
        system_model = SystemEnergyModel()
        system_model.build_energy_model(full_df)
        PersistenceManager.save_model(system_model, SYSTEM_ENERGY_MODEL_FILE_NAME)

    print("~~~~~~~~~~~~ Finished building System Energy Model ~~~~~~~~~~~~")
    print("\n\n~~~~~~~~~~~~ Starting building Process Energy Model ~~~~~~~~~~~~")
    if PersistenceManager.model_exists(PROCESS_ENERGY_MODEL_FILE_NAME):
        process_energy_model = PersistenceManager.load_model(PROCESS_ENERGY_MODEL_FILE_NAME)
    else:
        process_energy_model = ProcessEnergyModel(system_model)
        process_energy_model.build_energy_model(full_df)
        PersistenceManager.save_model(process_energy_model, PROCESS_ENERGY_MODEL_FILE_NAME)

    print("Done, you can use the energy model now :)")
