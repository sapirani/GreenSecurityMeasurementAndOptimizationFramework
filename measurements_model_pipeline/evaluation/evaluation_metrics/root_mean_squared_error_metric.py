import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class RootMeanSquaredErrorMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "Root Mean Squared Error (RMSE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        return math.sqrt(mean_squared_error(y, y_pred))
