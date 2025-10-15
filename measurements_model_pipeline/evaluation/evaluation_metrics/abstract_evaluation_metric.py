from abc import ABC, abstractmethod

import pandas as pd


class AbstractEvaluationMetric(ABC):
    @abstractmethod
    def get_metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        pass