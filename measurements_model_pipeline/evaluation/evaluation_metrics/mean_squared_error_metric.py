import pandas as pd
from sklearn.metrics import mean_squared_error

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class MeanSquaredErrorMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "Mean Squared Error (MSE)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        return mean_squared_error(y_pred, y)
