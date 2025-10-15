import pandas as pd

from measurements_model_pipeline.evaluation.evaluation_metrics.abstract_evaluation_metric import \
    AbstractEvaluationMetric


def evaluate_performance(y_true: pd.Series, y_pred: pd.Series, metrics: list[AbstractEvaluationMetric]) -> dict[str, float]:
    metrics_results = {}
    for metric in metrics:
        metrics_results[metric.get_metric_name()] = metric.evaluate(y_true, y_pred)

    return metrics_results