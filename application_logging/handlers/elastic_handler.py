import time
from datetime import datetime, timezone
import logging
import os
from elasticsearch import Elasticsearch

INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "scanner")


class ElasticSearchLogHandler(logging.Handler):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str = INDEX_NAME,
            start_timestamp: float = time.time()
    ):
        super().__init__()
        self.es = Elasticsearch(elastic_url, basic_auth=(elastic_username, elastic_password))
        self.index_name = index_name
        self.start_date = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()

        if not self.es.ping():
            print("Cannot connect to Elastic")
            raise ConnectionError("Elasticsearch cluster is not reachable")

    def emit(self, record: logging.LogRecord):
        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "start_date": self.start_date
        }

        # Emit extra log data
        reserved = set(vars(logging.makeLogRecord({})).keys())
        for key, value in record.__dict__.items():
            if key not in reserved:
                doc[key] = value

        self.es.index(index=self.index_name, body=doc)
