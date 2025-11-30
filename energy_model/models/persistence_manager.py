import os

import joblib

from energy_model.models.abstract_energy_model import AbstractEnergyModel


class PersistenceManager:
    @staticmethod
    def model_exists(path: str) -> bool:
        return os.path.isfile(path)

    @staticmethod
    def save_model(model: AbstractEnergyModel, path: str):
        joblib.dump(model, path)

    @staticmethod
    def load_model(path: str) -> AbstractEnergyModel:
        return joblib.load(path)
