from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from measurements_model.dataset_processing.feature_selection.all_features_no_network import AllFeaturesNoNetwork


class BestModelConfig:
    MODEL_NAME = RandomForestRegressor
    MODEL_PARAMETERS = {
        "n_estimators": 500,
        "max_features": 'sqrt',
        "max_depth": 7,
        "min_samples_split": 5
    }


KB = 1000
ENERGY_CONSUMPTION_PER_KBYTE_SENT = 0.00587
ENERGY_CONSUMPTION_PER_KBYTE_RECEIVED = 0.1161


class MeasurementsModel:
    def __init__(self, network_sent_bytes_column: Optional[str], network_received_bytes_column: Optional[str]):
        self.__model = BestModelConfig.MODEL_NAME(**BestModelConfig.MODEL_PARAMETERS)
        self.__network_sent_bytes_column = network_sent_bytes_column
        self.__network_received_bytes_column = network_received_bytes_column
        self.__feature_selector_no_network = AllFeaturesNoNetwork()

    def fit(self, X, y):
        y = np.array(y) + self.__calculate_total_network_energy_usage(X)
        X = self.__feature_selector_no_network.select_features(X)
        self.__model.fit(X, y)

    def __calculate_network_energy_usage(self, row) -> float:
        network_energy_addition = 0
        if self.__network_sent_bytes_column in row:
            network_energy_addition += row[self.__network_sent_bytes_column] * KB * ENERGY_CONSUMPTION_PER_KBYTE_SENT

        if self.__network_received_bytes_column in row:
            network_energy_addition += row[
                                           self.__network_received_bytes_column] * KB * ENERGY_CONSUMPTION_PER_KBYTE_RECEIVED

        return network_energy_addition

    def __calculate_total_network_energy_usage(self, X):
        if (self.__network_sent_bytes_column is None and self.__network_received_bytes_column is None) or \
                (self.__network_sent_bytes_column not in X and self.__network_received_bytes_column not in X):
            return np.zeros(len(X))

        return np.array([self.__calculate_network_energy_usage(row) for row in X])

    def predict(self, X):
        X_without_network = self.__feature_selector_no_network.select_features(X)
        y_pred = self.__model.predict(X_without_network)
        y_pred = np.array(y_pred) + self.__calculate_total_network_energy_usage(X)
        return y_pred
