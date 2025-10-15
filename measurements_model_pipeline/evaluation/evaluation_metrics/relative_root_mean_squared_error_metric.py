import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from measurements_model_pipeline.evaluation.evaluation_metrics.root_mean_squared_error_metric import \
    RootMeanSquaredErrorMetric


class RelativeRootMeanSquaredErrorMetric(RootMeanSquaredErrorMetric):
    def get_metric_name(self) -> str:
        return "Relative Root Mean Squared Error (Relative RMSE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        rmse = super().evaluate(y, y_pred)
        y_average = y.mean()
        return (rmse / y_average) * 100
