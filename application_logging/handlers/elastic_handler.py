import time
from datetime import datetime, timezone
import logging
import os
from logging import Handler

from elasticsearch import Elasticsearch

INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "scanner")


def get_elastic_logging_handler(
        elastic_username: str,
        elastic_password: str,
        elastic_url: str,
        index_name: str,
        starting_time: float = time.time()
) -> Handler:
    try:
        return ElasticSearchLogHandler(
            elastic_username=elastic_username,
            elastic_password=elastic_password,
            elastic_url=elastic_url,
            index_name=index_name,
            start_timestamp=starting_time
        )
    except ConnectionError:
        return None


class ElasticSearchLogHandler(logging.Handler):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str,
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

        self.es.index(index=self.index_name, body=doc)
