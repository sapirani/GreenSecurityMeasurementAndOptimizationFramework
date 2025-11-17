import threading

import pandas as pd

from energy_model.energy_model_parameters import PROCESS_ENERGY_MODEL_FILE_NAME
from energy_model.models.persistence_manager import PersistenceManager
from energy_model.models.process_energy_model import ProcessEnergyModel


class AggregationsEnergyModel:
    """
    This class is the measurement model after loading it from a pickle file.
    The model (after loading) should be of type 'EnergyPredictionModel', after calling 'fit' and training the model.
    """
    __instance = None
    __lock = threading.Lock()  # for thread safe - maybe unnecessary
    __model: ProcessEnergyModel = None

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
                    if PersistenceManager.model_exists(PROCESS_ENERGY_MODEL_FILE_NAME):
                        self.__model = PersistenceManager.load_model(PROCESS_ENERGY_MODEL_FILE_NAME)
                    else:
                        raise RuntimeError(f"Model file {PROCESS_ENERGY_MODEL_FILE_NAME} does not exist, build the model first.")

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
