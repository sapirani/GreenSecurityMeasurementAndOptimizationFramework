import pandas as pd


def print_scores_per_metric(scores: dict[str, float]):
    print("*** Model's Accuracy on test-set ***")
    for metric, score in scores.items():
        print(f"{metric} value: {score}")

def evaluate_performance(y_true: pd.Series, y_pred: pd.Series, metrics: list[AbstractEvaluationMetric]) -> dict[str, float]:
    metrics_results = {}
    for metric in metrics:
        metrics_results[metric.get_metric_name()] = metric.evaluate(y_true, y_pred)

    return metrics_results

def calculate_and_print_scores(y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    scores_per_metric = evaluate_performance(y, y_pred, EVALUATION_METRICS)
    print_scores_per_metric(scores_per_metric)
    return scores_per_metric