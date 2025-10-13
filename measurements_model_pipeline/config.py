import os
from pathlib import Path

from measurements_model_pipeline.evaluation.evaluation_metrics.average_per_metric import AveragePERMetric
from measurements_model_pipeline.evaluation.evaluation_metrics.mean_absolute_error_metric import MeanAbsoluteErrorMetric
from measurements_model_pipeline.evaluation.evaluation_metrics.mean_squared_error_metric import MeanSquaredErrorMetric
from measurements_model_pipeline.evaluation.evaluation_metrics.relative_root_mean_squared_error_metric import \
    RelativeRootMeanSquaredErrorMetric
from measurements_model_pipeline.evaluation.evaluation_metrics.root_mean_squared_error_metric import \
    RootMeanSquaredErrorMetric
from measurements_model_pipeline.evaluation.evaluation_metrics.standard_deviation_metric import StandardDeviationMetric

SCORING_METHODS_FOR_MODEL = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
EVALUATION_METRICS = [RelativeRootMeanSquaredErrorMetric(), StandardDeviationMetric(),
                      AveragePERMetric(), MeanSquaredErrorMetric(),
                      RootMeanSquaredErrorMetric(), MeanAbsoluteErrorMetric()]

BASE_MODEL_DIR = Path(__file__).parent
MODEL_FILE_NAME = os.path.join(BASE_MODEL_DIR, r"energy_prediction_model.pkl")
