from logging import StreamHandler

from application_logging.logging_utils import get_measurement_logger
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from application_logging.formatters.pretty_extra_formatter import PrettyExtrasFormatter
from utils.general_consts import LoggerName, IndexName

ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SwmQNU7y"
INDEX_SYSTEM = "system_metrics"
INDEX_PROCESS = "process_metrics"
PULL_INTERVAL_SECONDS = 2  # seconds
MAX_INDEXING_TIME_SECONDS = 15
PULL_PAGE_SIZE = 10000


logger = get_measurement_logger(
    logger_name=LoggerName.METRICS_AGGREGATIONS,
    logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS, pipeline_name="search_name_enrich"),
)

# handler = StreamHandler()
# handler.setFormatter(PrettyExtrasFormatter())
# logger.addHandler(handler)
