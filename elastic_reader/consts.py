from enum import Enum


class ElasticIndex(str, Enum):
    SYSTEM = "system_metrics"
    PROCESS = "process_metrics"
    AGGREGATIONS = "metrics_aggregations"


class ElasticConsumerType(str, Enum):
    DRL = "DRL"
    CSV = "CSV"
    AGGREGATIONS_LOGGER = "AGGREGATIONS_LOGGER"


class Verbosity(str, Enum):
    VERBOSE = "verbose"
    NONE = "none"


class TimePickerInputStrategy(str, Enum):
    GUI = "GUI"
    CLI = "CLI"
    FROM_CONFIGURATION = "from_configuration"


class AggregationStrategy(str, Enum):
    PULL_FROM_ELASTIC = "pull"  # TODO: SUPPORT
    CALCULATE = "calculate"
    NONE = "none"


FINAL_ITERATION_TIMEOUT_SECONDS = 120
PULL_INTERVAL_SECONDS = 5  # seconds
MAX_INDEXING_TIME_SECONDS = 15
PULL_PAGE_SIZE = 10000
