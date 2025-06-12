import datetime
import logging
import os
from program_parameters import elastic_username, elastic_url, elastic_password
from elasticsearch import Elasticsearch

INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "scanner")


# TODO: remove the program_parameters dependency and start relying on parameters passed to the constructor
class ElasticSearchLogHandler(logging.Handler):
    def __init__(self, es_host: str = elastic_url, index_name: str = INDEX_NAME):
        super().__init__()
        self.es = Elasticsearch(es_host, basic_auth=(elastic_username, elastic_password))
        self.index_name = index_name
        self.start_date = datetime.datetime.utcnow().isoformat()

        if not self.es.ping():
            print("Cannot connect to Elastic")
            raise ConnectionError("Elasticsearch cluster is not reachable")

    def emit(self, record: logging.LogRecord) -> None:
        doc = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
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

        self.es.index(index=self.index_name, body=doc)
