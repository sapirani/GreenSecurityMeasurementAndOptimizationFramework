import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from measurements_model_pipeline.evaluation.evaluation_metrics.root_mean_squared_error_metric import \
    RootMeanSquaredErrorMetric


class RelativeRootMeanSquaredErrorMetric(RootMeanSquaredErrorMetric):
    def get_metric_name(self) -> str:
        return "Relative RMSE"

    def calculate_metric(self, y: pd.Series, y_pred: pd.Series) -> float:
        mse = mean_squared_error(y_pred, y)
        relative_den = (y_pred ** 2).sum()
        relative_mse = mse / relative_den
        return math.sqrt(relative_mse)
