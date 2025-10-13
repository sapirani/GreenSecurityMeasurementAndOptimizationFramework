import pandas as pd

from measurements_model_pipeline.config import EVALUATION_METRICS
from measurements_model_pipeline.evaluation.evaluation_metrics_utils import evaluate_performance


def print_scores_per_metric(scores: dict[str, float]):
    print("*** Model's Accuracy on test-set ***")
    for metric, score in scores.items():
        print(f"{metric} value: {score}")

def calculate_and_print_scores(y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    # todo: ADD MORE METRICS
    scores_per_metric = evaluate_performance(y, y_pred, EVALUATION_METRICS)
    print_scores_per_metric(scores_per_metric)
    return scores_per_metric

if __name__ == "__main__":
    y = pd.Series([1, 2, 3, 4, 5])
    y_pred1 = pd.Series([1, 2, 3, 4, 5])
    y_pred2 = pd.Series([2, 4, 6, 8, 10])

    calculate_and_print_scores(y, y_pred2)