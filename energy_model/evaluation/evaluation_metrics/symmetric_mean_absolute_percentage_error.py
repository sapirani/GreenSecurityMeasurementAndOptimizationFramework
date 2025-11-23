import numpy as np
import pandas as pd

from energy_model.evaluation.evaluation_metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class SymmetricMeanAbsolutePercentageError(AbstractEvaluationMetric):
    """
    This class implements the symmetric mean absolute percentage error metric.
    Given the actual values y and the predicted values y_hat, the SMAPE is calculated as:
    - the average of the absolute percentage errors between actual values y and the predicted values y_hat
    - each error is weighted by the sum of the absolute values of the actual and predicted values.

    This metric should better present the errors when the y values are near 0.
    """
    def get_metric_name(self) -> str:
        return "Symmetric Mean Absolute Percentage Error (Symmetric MAPE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        smape = 1 / len(y) * np.sum(2 * np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred)) * 100)
        return smape
