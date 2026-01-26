import time
from abc import abstractmethod, ABC
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any
from elasticsearch import Elasticsearch

TIMESTAMP_FIELD_NAME = "timestamp"
START_DATE_FIELD_NAME = "start_date"


class AbstractElasticSearchHandler(logging.Handler, ABC):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str,
            start_timestamp: Optional[float] = None,
            pipeline_name: Optional[str] = None
    ):
        super().__init__()
        self.es = Elasticsearch(elastic_url, basic_auth=(elastic_username, elastic_password))
        self.index_name = index_name
        self.start_date = None
        if start_timestamp:
            self.start_date = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()
        self.pipeline_name = pipeline_name

        if not self.es.ping():
            print("Cannot connect to Elastic")
            raise ConnectionError("Elasticsearch cluster is not reachable")

    @abstractmethod
    def _inner_emit(self, doc: Dict[str, Any]):
        pass

    def emit(self, record: logging.LogRecord):
        doc = {
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Emit extra log data
        reserved = set(vars(logging.makeLogRecord({})).keys())
        for key, value in record.__dict__.items():
            if key not in reserved:
                doc[key] = value

        if TIMESTAMP_FIELD_NAME not in doc:
            doc[TIMESTAMP_FIELD_NAME] = datetime.now(timezone.utc).isoformat()

        # TODO: try to find a way to avoid sending start_date inside each log
        if self.start_date:
            doc[START_DATE_FIELD_NAME] = self.start_date

        self._inner_emit(doc)
