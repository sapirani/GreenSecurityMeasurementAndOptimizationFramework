import logging


def setup_logging():
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
