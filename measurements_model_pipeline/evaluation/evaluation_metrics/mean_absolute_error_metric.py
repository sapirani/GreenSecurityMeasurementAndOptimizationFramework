import pandas as pd
from sklearn.metrics import mean_absolute_error

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class MeanAbsoluteErrorMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "MAE"
    def calculate_metric(self, y: pd.Series, y_pred: pd.Series) -> float:
        return mean_absolute_error(y_pred, y)