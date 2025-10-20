import numpy as np
import pandas as pd

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class AveragePERMetric(AbstractEvaluationMetric):
    """
    This class computes the average of PER metric for each sample.
    It measures, in percentage, how far the model's predictions (y_pred) are from the true values (y) — relative to the magnitude of the true values.
    It tells the average deviation in percentage from the actual (true) value.
    A lower PER means your predictions are closer to reality.
    A PER = 0% means a perfect prediction.
    """
    def get_metric_name(self) -> str:
        return "Average Percentage Error Rate (Average PER)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)

        mask = y != 0
        y = y[mask]
        y_pred = y_pred[mask]

        return np.mean(abs(y - y_pred) / y) * 100