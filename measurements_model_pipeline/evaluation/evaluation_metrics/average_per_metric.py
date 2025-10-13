import pandas as pd

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class AveragePERMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "Average PER"

    def calculate_metric(self, y: pd.Series, y_pred: pd.Series) -> float:
        return (abs(y - y_pred) / y).mean() * 100