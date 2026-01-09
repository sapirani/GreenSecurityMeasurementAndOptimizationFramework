from typing import Any, Union, Callable
from sklearn.model_selection import train_test_split
import pandas as pd

from energy_model.configs.defaults_configs import DEFAULT_SCORING_METHODS_GRID_SEARCH
from energy_model.evaluation.grid_search.model_selector import ModelSelector


class GridSearchPipelineExecutor:
    def __init__(self, possible_models: list[dict], scoring_methods: dict[str, Union[str, Callable]] = None):
        self.__possible_models = possible_models
        if scoring_methods is None:
            scoring_methods = DEFAULT_SCORING_METHODS_GRID_SEARCH
        self.__evaluation_methods = scoring_methods

    def run_grid_search(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.copy()
        X_test = X_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()

        model_selector = ModelSelector(models_to_experiment=self.__possible_models)
        best_model_per_metric = model_selector.choose_best_model(self.__evaluation_methods, X_train, X_test, y_train, y_test)
        return best_model_per_metric
