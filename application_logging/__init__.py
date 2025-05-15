import logging
import threading
from typing import Optional

from application_logging.handlers.elastic_handler import ElasticSearchLogHandler

_logger = None
_logger_lock = threading.Lock()
_measurement_session_id: Optional[str] = None


def set_measurement_session_id(measurement_session_id: str) -> None:
    global _measurement_session_id
    _measurement_session_id = measurement_session_id


def get_measurement_logger() -> logging.Logger:
    if _measurement_session_id is None:
        raise ValueError("measurement session id must be set before calling the logger retrieval function")

    global _logger
    with _logger_lock:
        if _logger is None:
            _logger = logging.getLogger("measurements_logger")
            _logger.setLevel(logging.INFO)

            handler = logging.NullHandler()
            try:
                handler = ElasticSearchLogHandler(_measurement_session_id)
            except ConnectionError:
                pass
            formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
            handler.setFormatter(formatter)

            _logger.addHandler(handler)

    return _logger
