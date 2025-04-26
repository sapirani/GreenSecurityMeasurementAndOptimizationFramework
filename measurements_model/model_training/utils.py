import math

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_scores_per_metric(scores: dict[str, float]):
    print("*** Model's Accuracy on test-set ***")
    for metric, score in scores.items():
        print(f"{metric} value: {score}")

def calculate_and_print_scores(y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    # todo: ADD MORE METRICS
    scores_per_metric = {}
    PER = (abs(y - y_pred) / y).mean() * 100
    scores_per_metric["Average PER"] = PER

    MSE = mean_squared_error(y_pred, y)
    scores_per_metric["MSE"] = MSE
    scores_per_metric["RMSE"] = math.sqrt(MSE)

    scores_per_metric["MAE"] = mean_absolute_error(y_pred, y)
    print_scores_per_metric(scores_per_metric)
    return scores_per_metric