from enum import Enum

ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"


class ElasticIndex(str, Enum):
    SYSTEM = "system_metrics"
    PROCESS = "process_metrics"
    AGGREGATIONS = "metrics_aggregations"


FINAL_ITERATION_TIMEOUT_SECONDS = 120
PULL_INTERVAL_SECONDS = 2  # seconds
MAX_INDEXING_TIME_SECONDS = 15
PULL_PAGE_SIZE = 10000
