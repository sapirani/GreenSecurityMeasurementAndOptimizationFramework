import pandas as pd

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


class StandardDeviationMetric(AbstractEvaluationMetric):
    def get_metric_name(self) -> str:
        return "Standard Deviation (Std.)"

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> float:
        return abs(y - y_pred).std()
