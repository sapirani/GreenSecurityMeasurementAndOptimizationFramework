import pandas as pd

from measurements_model.config import ALL_MEASUREMENTS_DIR_PATH, IDLE_DIR_PATH
from measurements_model.dataset_creation.dataset_creator import DatasetCreator
from measurements_model.energy_model import EnergyModel

if __name__ == "__main__":
    creator = DatasetCreator(
        sessions_dir=ALL_MEASUREMENTS_DIR_PATH,
        idle_session_path=IDLE_DIR_PATH
    )

    df = creator.create_dataset()



    model = EnergyModel()
    model.initialize_model()
    model.predict(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))