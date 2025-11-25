import numpy as np
import pandas as pd

from energy_model.evaluation.evaluation_metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class SymmetricMeanAbsolutePercentageError(AbstractEvaluationMetric):
    """
    This class implements the symmetric mean absolute percentage error metric.
    Given the actual values y and the predicted values y_hat, the SMAPE is calculated as:
    - the average of the absolute percentage errors between actual values y and the predicted values y_hat
    - each error is weighted by the magnitude of the prediction and the target combined.

    The denominator is the average of y and y_hat - in order to deal with y ~ 0 and big values of y_hat.
    why does it work like this?
        * Using only the true value breaks when y is small or zero
        * Using only prediction also unfair
        * Hence, using the average treats both equally and avoids explosions

    This metric should better present the errors when the y values are near 0.

    For example, if SMAPE = 20% means that the model’s predictions differ from the true values by 20% on average relative to the magnitude of the prediction and the target combined.
    """
    def get_metric_name(self) -> str:
        return "Symmetric Mean Absolute Percentage Error (Symmetric MAPE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        smape = 1 / len(y) * np.sum(2 * np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred)) * 100)
        return smape
