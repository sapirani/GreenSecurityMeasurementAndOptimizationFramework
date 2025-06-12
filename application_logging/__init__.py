import logging
from typing import Protocol, Dict, Any, Optional

from application_logging.handlers.elastic_handler import ElasticSearchLogHandler


class AdapterFactoryProtocol(Protocol):
    keywords: Dict[str, Any]
    def __call__(self, logger: logging.Logger) -> logging.LoggerAdapter: ...


def get_measurement_logger(
        adapter_factory: AdapterFactoryProtocol,
        logger_handler: Optional[logging.Handler]
) -> logging.LoggerAdapter:

    if "session_id" not in adapter_factory.keywords.get('extra', {}):
        raise ValueError("'session_id must' be inserted into the adapter_factory before calling this function")

    _logger = logging.getLogger("measurements_logger")
    _logger.setLevel(logging.INFO)

    if not _logger.handlers:
        handler = logger_handler if logger_handler else logging.NullHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    adapter = adapter_factory(_logger)
    return adapter
