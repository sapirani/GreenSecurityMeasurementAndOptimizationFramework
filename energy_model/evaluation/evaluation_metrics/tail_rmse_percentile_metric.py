import numpy as np
import pandas as pd

from energy_model.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class TailRootMeanSquaredErrorMetric(AbstractEvaluationMetric):
    """
    Tail RMSE (p%):
    Measures the root mean squared error computed only on the worst p% of prediction errors.
    Specifically, it selects the p-th percentile of squared errors and computes RMSE over errors
    greater than or equal to this threshold.

    This metric emphasizes worst-case behavior and penalizes large, rare errors that are often
    hidden by average-based metrics such as RMSE or MSE.

    Lower values indicate better control over extreme prediction err
    """
    def __init__(self, percentile: int = 95) -> None:
        self._percentile = percentile

    def get_metric_name(self) -> str:
        return f"Tail Root Mean Squared Error Percentile {self._percentile} (Tail-RMSE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        y_true = np.asarray(y)
        y_pred = np.asarray(y_pred)

        squared_errors = (y_true - y_pred) ** 2
        threshold = np.percentile(squared_errors, self._percentile)

        tail_errors = squared_errors[squared_errors >= threshold]

        return float(np.sqrt(np.mean(tail_errors)))
