import logging

from application_logging import ElasticSearchLogHandler


def setup_logging() -> None:
    # Create a handler for printing to console
    console_handler = logging.StreamHandler()

    # Set the level for the handler to DEBUG (to capture all messages)
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(levelname)s: [%(asctime)s] - %(message)s')
    console_handler.setFormatter(formatter)

    # Get the root logger and add the console handler
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all log levels
    logger.addHandler(console_handler)


def get_elastic_logger(measurement_session_id: str) -> logging.Logger:
    elastic_logger = logging.getLogger("elastic_logger")
    elastic_logger.setLevel(logging.INFO)

    elastic_handler = ElasticSearchLogHandler(measurement_session_id)
    elastic_logger.addHandler(elastic_handler)

    return elastic_logger
