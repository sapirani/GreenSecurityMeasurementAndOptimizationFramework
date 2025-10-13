import pandas as pd

from measurements_model_pipeline.config import EVALUATION_METRICS
from measurements_model_pipeline.evaluation.evaluation_metrics_utils import evaluate_performance


def print_scores_per_metric(scores: dict[str, float]):
    print("*** Model's Accuracy on test-set ***")
    for metric, score in scores.items():
        print(f"{metric} value: {score}")


def calculate_and_print_scores(y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    scores_per_metric = evaluate_performance(y, y_pred, EVALUATION_METRICS)
    print_scores_per_metric(scores_per_metric)
    return scores_per_metric
