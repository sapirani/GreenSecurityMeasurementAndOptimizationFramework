import numpy as np
import pandas as pd

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class RootMeanSquaredPercentErrorMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "Root Mean Squared Percent Error (RMSPE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)

        # avoid division by zero
        mask = y != 0
        y = y[mask]
        y_pred = y_pred[mask]

        percentage_errors = ((y - y_pred) / y) ** 2
        return np.sqrt(np.mean(percentage_errors)) * 100