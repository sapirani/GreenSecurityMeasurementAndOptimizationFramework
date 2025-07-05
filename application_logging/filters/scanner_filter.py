import logging
from typing import Dict, Any


class ScannerLoggerFilter(logging.Filter):
    """
    This class actually injects custom fields that will be injected to each log
    """
    def __init__(self, session_id: str, hostname: str, user_defined_extras: Dict[str, Any]):
        super().__init__()

        self.additional_constant_extras = {"session_id": session_id, "hostname": hostname, **user_defined_extras}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.additional_constant_extras.items():
            setattr(record, key, value)

        return True
