import os
import threading

import joblib
import pandas as pd

from energy_model.models.energy_prediction_model import EnergyPredictionModel

ENERGY_MODEL_PATH = r"../energy_model.pkl"

class AggregationsEnergyModel:
    """
    This class is the measurement model after loading it from a pickle file.
    The model (after loading) should be of type 'EnergyPredictionModel', after calling 'fit' and training the model.
    """
    __instance = None
    __lock = threading.Lock()  # for thread safe - maybe unnecessary
    __model: EnergyPredictionModel = None

    def __init__(self):
        raise RuntimeError("This is a Singleton. Invoke get_instance() instead.")

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)

        return cls.__instance

    def initialize_model(self):
        if self.__model is None:
            with self.__lock:
                if self.__model is None:
                    if os.path.exists(ENERGY_MODEL_PATH):
                        self.__model = joblib.load(ENERGY_MODEL_PATH)
                    else:
                        raise RuntimeError(f"Model file {ENERGY_MODEL_PATH} does not exist, build the model first.")

    def predict(self, samples: pd.DataFrame) -> list[float]:
        """
        This method predicts the energy of the given samples using the energy prediction model.
        :param samples: DataFrame with samples to predict their energy.
        :return: Predicted energy for each sample.
        """
        predictions = self.__model.predict(samples)
        if len(predictions) > 0:
            return predictions.tolist()
        else:
            raise RuntimeError(f"No predictions found for samples {samples}.")