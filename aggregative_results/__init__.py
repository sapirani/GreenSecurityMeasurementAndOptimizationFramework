from logging import StreamHandler

from application_logging import get_measurement_logger, get_elastic_logging_handler
from application_logging.formatters.pretty_extra_formatter import PrettyExtrasFormatter
from utils.general_consts import LoggerName, IndexName

ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"
INDEX_SYSTEM = "system_metrics"
INDEX_PROCESS = "process_metrics"
PULL_INTERVAL_SECONDS = 2  # seconds


logger = get_measurement_logger(
    logger_name=LoggerName.METRICS_AGGREGATIONS,
    logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS),
)

handler = StreamHandler()
handler.setFormatter(PrettyExtrasFormatter())
logger.addHandler(handler)
