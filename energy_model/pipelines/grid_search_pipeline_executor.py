from typing import Any
from sklearn.model_selection import train_test_split
import pandas as pd

from energy_model.configs.defaults_configs import DEFAULT_SCORING_METHODS_GRID_SEARCH
from energy_model.evaluation.grid_search.model_selector import ModelSelector
from energy_model.evaluation.grid_search.models_config import REGRESSION_MODELS_WITH_PARAMETERS


class GridSearchPipelineExecutor:
    def __init__(self, possible_models: list[dict] = REGRESSION_MODELS_WITH_PARAMETERS):
        self.__possible_models = possible_models
        self.__evaluation_methods = DEFAULT_SCORING_METHODS_GRID_SEARCH

    def run_grid_search(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.copy()
        X_test = X_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()

        model_selector = ModelSelector(models_to_experiment=self.__possible_models)
        best_model = model_selector.choose_best_model(self.__evaluation_methods, X_train, X_test, y_train, y_test)
        return best_model
