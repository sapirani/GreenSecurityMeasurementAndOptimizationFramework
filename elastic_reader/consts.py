from enum import Enum


class IndexRetrievalOrder:
    FIRST = 1
    SECOND = 2
    THIRD = 3
    LAST = 999


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
MAX_INDEXING_TIME_SECONDS = 16
PULL_PAGE_SIZE = 10000
