import os

import joblib
import pandas as pd

from measurements_model.main_model_configuration import MODEL_FILE_NAME


class EnergyModel:
    def __init__(self):
        self.__model = None

    def initialize_model(self):
        if os.path.exists(MODEL_FILE_NAME):
            self.__model = joblib.load(MODEL_FILE_NAME)
        else:
            raise RuntimeError(f"Model file {MODEL_FILE_NAME} does not exist, build the model first.")

    def predict(self, sample: pd.DataFrame) -> float:
        return self.__model.predict(sample)
