import threading
from enum import Enum

import pandas as pd

from energy_model.energy_model_parameters import PROCESS_ENERGY_MODEL_FILE_NAME, SYSTEM_ENERGY_MODEL_FILE_NAME
from energy_model.models.abstract_energy_model import AbstractEnergyModel
from energy_model.models.persistence_manager import PersistenceManager


class ModelType(Enum, str):
    SystemBased = "System-Model"
    ProcessBased = "Process-Model"


class AggregationsEnergyModel:
    """
    This class is the measurement model after loading it from a pickle file.
    The model (after loading) should be of type 'EnergyPredictionModel', after calling 'fit' and training the model.
    """
    __instance = None
    __lock = threading.Lock()  # for thread safe - maybe unnecessary
    __models: dict[ModelType, AbstractEnergyModel] = {}

    def __init__(self):
        raise RuntimeError("This is a Singleton. Invoke get_instance() instead.")

    @classmethod
    def get_energy_model_instance(cls, model_type: ModelType) -> AbstractEnergyModel:
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)

        model = cls.__initialize_model(model_type)
        return model

    @classmethod
    def __initialize_model(cls, model_type: ModelType) -> AbstractEnergyModel:
        with cls.__lock:
            if model_type not in cls.__models.keys():
                model = None
                if model_type == ModelType.ProcessBased and PersistenceManager.model_exists(
                        PROCESS_ENERGY_MODEL_FILE_NAME):
                    model = PersistenceManager.load_model(PROCESS_ENERGY_MODEL_FILE_NAME)
                elif model_type == ModelType.SystemBased and PersistenceManager.model_exists(
                        SYSTEM_ENERGY_MODEL_FILE_NAME):
                    model = PersistenceManager.load_model(SYSTEM_ENERGY_MODEL_FILE_NAME)

                if model:
                    cls.__models[model_type] = model
                else:
                    raise RuntimeError(
                        f"Model file for model of type {model_type.value} does not exist, build the model first.")
        return model

    def predict(self, samples: pd.DataFrame, model_type: ModelType) -> list[float]:
        """
        This method predicts the energy of the given samples using the energy prediction model.
        :param samples: DataFrame with samples to predict their energy.
        :param model_type: Type of model to use (Process or System).
        :return: Predicted energy for each sample.
        """
        with self.__lock:
            if model_type not in self.__models.keys():
                raise RuntimeError("The model has not been initialized. Call initialize_model() first.")

            model = self.__models[model_type]

        predictions = model.predict(samples)
        if len(predictions) > 0:
            return predictions.tolist()
        else:
            raise RuntimeError(f"No predictions found for samples {samples}.")
