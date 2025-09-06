from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from measurements_model.dataset_processing.feature_selection.process_and_full_system_feature_selector import \
    ProcessAndTotalSystem


class BestModelConfig:
    MODEL_NAME = HistGradientBoostingRegressor
    MODEL_PARAMETERS = {
        "max_depth": 11,
        "min_samples_leaf": 10
    }


class MeasurementsModel:
    def __init__(self):
        self.__model = BestModelConfig.MODEL_NAME(**BestModelConfig.MODEL_PARAMETERS)
        self.__feature_selector_no_network = ProcessAndTotalSystem()

    def fit(self, X, y):
        y = np.array(y)
        X = self.__feature_selector_no_network.select_features(X)
        self.__model.fit(X, y)

    def predict(self, X):
        X_without_network = self.__feature_selector_no_network.select_features(X)
        y_pred = self.__model.predict(X_without_network)
        y_pred = np.array(y_pred)
        return y_pred
