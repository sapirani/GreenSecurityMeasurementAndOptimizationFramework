from energy_model.configs.columns import SystemColumns
from energy_model.dataset_processing.filters.negative_value_filter import NegativeValueFilter
from energy_model.dataset_processing.filters.outliers_filter import OutlierFilter
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

DEFAULT_BEST_MODEL_METRIC = "Mean Absolute Error (MAE)"

NON_NEGATIVE_COLUMNS = [SystemColumns.DURATION_COL, SystemColumns.DISK_READ_BYTES_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.CPU_SYSTEM_COL]
OUTLIERS_COLUMNS = [SystemColumns.MEMORY_SYSTEM_COL, SystemColumns.DISK_READ_BYTES_SYSTEM_COL,
                    SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.CPU_SYSTEM_COL]
DEFAULT_FILTERS = [NegativeValueFilter(NON_NEGATIVE_COLUMNS),
                   OutlierFilter(OUTLIERS_COLUMNS)]

DEFAULT_SCORING_METHODS_GRID_SEARCH = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

DEFAULT_TRAIN_TEST_RATIO = 0.2
DEFAULT_CV_SPLITS_N = 5