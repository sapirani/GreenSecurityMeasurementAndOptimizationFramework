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
            Computes a percentile of the squared prediction errors.

            This metric measures the squared error between true and predicted values and
            returns the specified percentile of those errors.
            Unlike Mean Squared Error (MSE), which averages all errors, this metric captures the behavior
            of a chosen portion of the error distribution (e.g., median error or worst-case tail errors).

            It is especially useful when robustness to outliers is desired or when focusing on high-error cases.
        """
        y_true = np.asarray(y)
        y_pred = np.asarray(y_pred)

        squared_errors = (y_true - y_pred) ** 2
        return float(np.percentile(squared_errors, self.__percentile))
