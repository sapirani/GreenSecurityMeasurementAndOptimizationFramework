from enum import Enum


# TODO: UNIFY WITH THE SAME ENUM CLASS FROM SCANNER'S GENERAL CONSTS
class ElasticIndex(str, Enum):
    SYSTEM = "system_metrics"
    PROCESS = "process_metrics"
    AGGREGATIONS = "metrics_aggregations"
    APPLICATION_FLOW = "application_flow"


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


NON_GRACEFUL_TERMINATION_DETECTION_WINDOW_SECONDS = 180
PULL_INTERVAL_SECONDS = 5  # seconds
MAX_INDEXING_TIME_SECONDS = 15
PULL_PAGE_SIZE = 10000
