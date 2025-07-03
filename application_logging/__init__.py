from logging import Handler, Logger
import logging
from typing import Optional

from application_logging.handlers.elastic_handler import ElasticSearchLogHandler


def get_elastic_logging_handler(elastic_username: str, elastic_password: str, elastic_url: str, starting_time: float) -> Optional[Handler]:
    try:
        return ElasticSearchLogHandler(elastic_username, elastic_password, elastic_url, start_timestamp=starting_time)
    except ConnectionError:
        return None


def get_measurement_logger(custom_filter: Optional[logging.Filter], logger_handler: Optional[Handler]) -> Logger:
    """
    :param custom_filter: a filter to apply on logs (may be used to insert dynamic fields to the logs)
    :param logger_handler: a handler to attach to the returned adapter (for example, ElasticSearchLogHandler)
    """
    _logger = logging.getLogger("measurements_logger")
    _logger.setLevel(logging.DEBUG)

    if not _logger.filters and custom_filter:
        _logger.addFilter(custom_filter)

    if not _logger.handlers and logger_handler:
        handler = logger_handler if logger_handler else logging.NullHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    return _logger
