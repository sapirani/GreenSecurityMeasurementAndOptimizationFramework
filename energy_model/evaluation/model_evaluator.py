import pandas as pd

from energy_model.configs.defaults_configs import DEFAULT_EVALUATION_METRICS
from energy_model.evaluation.evaluation_metrics.abstract_evaluation_metric import AbstractEvaluationMetric

class ModelEvaluator:
    def __init__(self, metrics: list[AbstractEvaluationMetric] = DEFAULT_EVALUATION_METRICS):
        self.__metrics = metrics

    def evaluate(self, y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        metrics_results = {}
        for metric in self.__metrics:
            metrics_results[metric.get_metric_name()] = metric.evaluate(y, y_pred)

        return metrics_results

    @staticmethod
    def print_results(metrics_results: dict[str, float]):
        print("*** Model's Accuracy on test-set ***")
        for metric, score in metrics_results.items():
            print(f"{metric} value: {score}")
