from typing import Any
from sklearn.model_selection import train_test_split
import pandas as pd

from energy_model.configs.defaults_configs import DEFAULT_SCORING_METHODS_GRID_SEARCH
from energy_model.evaluation.grid_search.model_selector import ModelSelector

from energy_model.evaluation.grid_search.models_config import REGRESSION_MODELS_WITH_PARAMETERS
from energy_model.pipelines.pipeline_utils import split_train_test


class GridSearchPipelineExecutor:
    def __init__(self):
        self.__possible_models = REGRESSION_MODELS_WITH_PARAMETERS
        self.__evaluation_methods = DEFAULT_SCORING_METHODS_GRID_SEARCH

    def run_grid_search(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = split_train_test(df, target_column)
        model_selector = ModelSelector(models_to_experiment=self.__possible_models)
        best_model = model_selector.choose_best_model(self.__evaluation_methods, X_train, X_test, y_train, y_test)
        return best_model
