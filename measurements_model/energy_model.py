
import joblib
import pandas as pd


MODEL_FILE_NAME = "energy_prediction_model.pkl"

class EnergyModel:
    def __init__(self):
        self.__model = joblib.load(MODEL_FILE_NAME)

    def predict(self, sample: pd.DataFrame) -> float:
        return self.__model.predict(sample)