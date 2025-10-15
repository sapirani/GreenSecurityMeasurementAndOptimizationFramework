import os
import threading

import joblib
import pandas as pd

from measurements_model_pipeline.config import MODEL_FILE_NAME


class EnergyModel:
    """
    This class is the measurement model after loading it from a pickle file.
    The model (after loading) should be of type 'MeasurememtModel', after calling 'fit' and training the model.
    """
    __instance = None
    __lock = threading.Lock()  # for thread safe - maybe unnecessary
    __model = None

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
                    if os.path.exists(MODEL_FILE_NAME):
                        self.__model = joblib.load(MODEL_FILE_NAME)
                    else:
                        raise RuntimeError(f"Model file {MODEL_FILE_NAME} does not exist, build the model first.")

    def predict(self, sample: pd.DataFrame) -> float:
        """
        This method uses the energy prediction model to predict the energy of a given sample.
        :param sample: A pandas dataframe with single row (the sample) to predict energy for.
        :return: The predicted energy for that specific sample.
        """
        predictions = self.__model.predict(sample)
        if len(predictions) > 0:
            return predictions[0]
        else:
            raise RuntimeError(f"No predictions found for sample {sample}.")