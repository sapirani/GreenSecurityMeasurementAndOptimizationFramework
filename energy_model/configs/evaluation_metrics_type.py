from enum import Enum


class EvaluationMetricType(str, Enum):
    SMAPE = "Symmetric Mean Absolute Percentage Error (Symmetric MAPE)"
    RRMSE = "Relative Root Mean Squared Error (Relative RMSE)"
    STD = "Standard Deviation (Std.)"
    AveragePER = "Average Percentage Error Rate (Average PER)"
    MSE = "Mean Squared Error (MSE)"
    RMSE = "Root Mean Squared Error (RMSE)"
    MAE = "Mean Absolute Error (MAE)"
    RMSPE = "Root Mean Squared Percent Error (RMSPE)"
    PercentileSquaredError90 = "Percentile Squared Error - percentile = 90"
    PercentileSquaredError95 = "Percentile Squared Error - percentile = 95"
    TailRMSE = "Tail Root Mean Squared Error Percentile 95 (Tail-RMSE)"
    RMSERatio = "Root Mean Squared Error Ratio for Percentile 95"
