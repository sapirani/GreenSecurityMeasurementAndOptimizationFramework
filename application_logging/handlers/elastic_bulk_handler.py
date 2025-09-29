import atexit
import time
from logging import Handler
from queue import Queue
from typing import Optional, Dict, Any
from elasticsearch import helpers
from application_logging.handlers.abstract_elastic_handler import AbstractElasticSearchHandler


def get_elastic_bulk_handler(
        elastic_username: str,
        elastic_password: str,
        elastic_url: str,
        index_name: str,
        starting_time: float = time.time(),
        pipeline_name: Optional[str] = None,
        max_queue_size: int = 10000
) -> Handler:
    try:
        return ElasticSearchBulkHandler(
            elastic_username=elastic_username,
            elastic_password=elastic_password,
            elastic_url=elastic_url,
            index_name=index_name,
            start_timestamp=starting_time,
            pipeline_name=pipeline_name,
            max_queue_size=max_queue_size
        )
    except ConnectionError:
        return None


class ElasticSearchBulkHandler(AbstractElasticSearchHandler):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str,
            start_timestamp: float = time.time(),
            pipeline_name: Optional[str] = None,
            max_queue_size: int = 10000
    ):
        super().__init__(elastic_username, elastic_password, elastic_url, index_name, start_timestamp, pipeline_name)
        self.queue = Queue()
        self.max_queue_size = max_queue_size
        atexit.register(self.flush)

    def _inner_emit(self, doc: Dict[str, Any]):
        self.queue.put(doc)

        if self.queue.qsize() >= self.max_queue_size:
            self.flush()

    def _drain_queue_gen(self) -> Dict[str, Any]:
        while not self.queue.empty():
            yield self.queue.get_nowait()

    def flush(self):
        """Send all queued logs to Elasticsearch"""
        actions = ({"_index": self.index_name, "_source": doc} for doc in self._drain_queue_gen())
        try:
            for success, info in helpers.streaming_bulk(self.es, actions, pipeline=self.pipeline_name):
                if not success:
                    print(f"Streaming bulk failed: {info}")
        except Exception as e:
            print(f"Elasticsearch flush failed: {e}")

    def close(self):
        """Flush remaining logs on close"""
        self.flush()
        super().close()
