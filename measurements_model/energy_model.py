import os

import joblib
import pandas as pd

from measurements_model.model_execution.main_model_configuration import MODEL_FILE_NAME


class EnergyModel:
    """
    This class is the measurement model after loading it from a pickle file.
    The model (after loading) should be of type 'MeasurememtModel', after calling 'fit' and training the model.
    """
    def __init__(self):
        self.__model = None

    def initialize_model(self):
        if os.path.exists(MODEL_FILE_NAME):
            self.__model = joblib.load(MODEL_FILE_NAME)
        else:
            raise RuntimeError(f"Model file {MODEL_FILE_NAME} does not exist, build the model first.")

    def predict(self, sample: pd.DataFrame) -> float:
        return self.__model.predict(sample)
