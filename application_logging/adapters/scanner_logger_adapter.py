import logging
from typing import Dict, Any


class ScannerLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, session_id: str, hostname: str, user_defined_extras: Dict[str, Any]):
        super().__init__(logger, extra={"session_id": session_id, "hostname": hostname, **user_defined_extras})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        # Merge adapter's `extra` with any per-call extras
        merged_extra = {**self.extra, **extra}
        kwargs["extra"] = merged_extra
        return msg, kwargs
