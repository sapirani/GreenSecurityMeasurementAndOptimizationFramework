from energy_model.evaluation.evaluation_metrics.average_per_metric import AveragePERMetric
from energy_model.evaluation.evaluation_metrics.mean_absolute_error_metric import MeanAbsoluteErrorMetric
from energy_model.evaluation.evaluation_metrics.mean_squared_error_metric import MeanSquaredErrorMetric
from energy_model.evaluation.evaluation_metrics.relative_root_mean_squared_error_metric import \
    RelativeRootMeanSquaredErrorMetric
from energy_model.evaluation.evaluation_metrics.root_mean_squared_error_metric import RootMeanSquaredErrorMetric
from energy_model.evaluation.evaluation_metrics.root_mean_squared_percent_error_metric import \
    RootMeanSquaredPercentErrorMetric
from energy_model.evaluation.evaluation_metrics.standard_deviation_metric import StandardDeviationMetric

DEFAULT_EVALUATION_METRICS = [RelativeRootMeanSquaredErrorMetric(), StandardDeviationMetric(),
                              AveragePERMetric(), MeanSquaredErrorMetric(),
                              RootMeanSquaredErrorMetric(), MeanAbsoluteErrorMetric(),
                              RootMeanSquaredPercentErrorMetric()]
