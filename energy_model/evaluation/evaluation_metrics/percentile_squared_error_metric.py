import numpy as np
import pandas as pd

from energy_model.evaluation.evaluation_metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class PercentileSquaredErrorMetric(AbstractEvaluationMetric):
    def __init__(self, percentile: int = 95) -> None:
        self.__percentile = percentile

    def get_metric_name(self) -> str:
        return f"Percentile Squared Error - percentile = {self.__percentile}"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        """
            Computes the percentile of squared errors.
        """
        y_true = np.asarray(y)
        y_pred = np.asarray(y_pred)

        squared_errors = (y_true - y_pred) ** 2
        return float(np.percentile(squared_errors, self.__percentile))
