import time
from abc import abstractmethod, ABC
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any
from elasticsearch import Elasticsearch


class AbstractElasticSearchHandler(logging.Handler, ABC):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str,
            start_timestamp: float = time.time(),
            pipeline_name: Optional[str] = None
    ):
        super().__init__()
        self.es = Elasticsearch(elastic_url, basic_auth=(elastic_username, elastic_password))
        self.index_name = index_name
        self.start_date = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()
        self.pipeline_name = pipeline_name

        if not self.es.ping():
            print("Cannot connect to Elastic")
            raise ConnectionError("Elasticsearch cluster is not reachable")

    @abstractmethod
    def _inner_emit(self, doc: Dict[str, Any]) -> None:
        pass

    def emit(self, record: logging.LogRecord):
        doc = {
            "level": record.levelname,
            "message": record.getMessage(),
            # TODO: try to find a way to avoid sending start_date inside each log
            "start_date": self.start_date
        }

        # Emit extra log data
        reserved = set(vars(logging.makeLogRecord({})).keys())
        for key, value in record.__dict__.items():
            if key not in reserved:
                doc[key] = value

        if "timestamp" not in doc:
            doc["timestamp"]: datetime.now(timezone.utc).isoformat()

        self._inner_emit(doc)
