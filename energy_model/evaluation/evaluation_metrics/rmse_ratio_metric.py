import pandas as pd

from energy_model.evaluation.evaluation_metrics.root_mean_squared_error_metric import RootMeanSquaredErrorMetric
from energy_model.evaluation.evaluation_metrics.tail_rmse_percentile_metric import TailRootMeanSquaredErrorMetric


class RootMeanSquaredErrorRatioMetric(TailRootMeanSquaredErrorMetric, RootMeanSquaredErrorMetric):
    """
    Tail-RMSE / RMSE Ratio:
    Measures the stability of the model by comparing its worst-case error magnitude
    (Tail RMSE) to its average error magnitude (RMSE).

    A lower ratio indicates a tighter error distribution and fewer catastrophic outliers.
    Higher values suggest that while the model performs well on average, it occasionally
    produces very large errors.

    This metric is scale-invariant and is especially useful when safety, robustness,
    or downstream model stability is more important than raw accuracy.
    """
    def get_metric_name(self) -> str:
        return f"Root Mean Squared Error Ratio for Percentile {self._percentile}"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        rmse = RootMeanSquaredErrorMetric.evaluate(self, y, y_pred)
        tail_rmse = TailRootMeanSquaredErrorMetric.evaluate(self, y, y_pred)
        return tail_rmse / rmse
