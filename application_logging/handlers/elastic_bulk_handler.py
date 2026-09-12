import atexit
import time
import traceback
from datetime import timezone, datetime
from logging import Handler
from queue import Queue
from typing import Optional, Dict, Any
from elasticsearch import helpers

from DTOs.logging.consts import IndexName
from application_logging.handlers.abstract_elastic_handler import AbstractElasticSearchHandler


def get_elastic_bulk_handler(
        elastic_username: str,
        elastic_password: str,
        elastic_url: str,
        index_name: str,
        starting_time: float = time.time(),
        pipeline_name: Optional[str] = None,
        max_queue_size: int = 10000,
        ignore_exceptions: bool = False,
        request_timeout: int = 10,
        max_retries: int = 0,
        initial_backoff: int = 2,
        max_backoff: int = 600
) -> Handler:
    try:
        return ElasticSearchBulkHandler(
            elastic_username=elastic_username,
            elastic_password=elastic_password,
            elastic_url=elastic_url,
            index_name=index_name,
            start_timestamp=starting_time,
            pipeline_name=pipeline_name,
            max_queue_size=max_queue_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff
        )
    except ConnectionError as e:
        if ignore_exceptions:
            return None
        else:
            raise e


class ElasticSearchBulkHandler(AbstractElasticSearchHandler):
    def __init__(
            self,
            elastic_username: str,
            elastic_password: str,
            elastic_url: str,
            index_name: str,
            start_timestamp: float = time.time(),
            pipeline_name: Optional[str] = None,
            max_queue_size: int = 10000,
            request_timeout: int = 10,
            max_retries: int = 0,
            initial_backoff: int = 2,
            max_backoff: int = 600
    ):
        super().__init__(
            elastic_username,
            elastic_password,
            elastic_url,
            index_name,
            start_timestamp,
            pipeline_name,
            request_timeout
        )
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
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
            for success, info in helpers.streaming_bulk(
                    self.es,
                    actions,
                    max_retries=self.max_retries,
                    initial_backoff=self.initial_backoff,
                    max_backoff=self.max_backoff,
                    pipeline=self.pipeline_name,
            ):
                if not success:
                    print(f"Streaming bulk failed: {info}")
        except Exception as e:
            print(f"Elasticsearch flush failed: {e}")
            try:
                response = self.es.index(
                    index=IndexName.APPLICATION_FLOW,
                    body={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "level": "ERROR",
                        "message": "Elasticsearch flush failed",
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "exception_repr": repr(e),
                        "exception_traceback": traceback.format_exc(),
                        "exception_status_code": getattr(e, "status_code", None),
                        "exception_body": getattr(e, "body", None),
                        "exception_info": getattr(e, "info", None),
                    },
                )
                print(f"Error successfully written to Elasticsearch: {response}")

            except Exception as log_error:
                print(
                    f"Failed to log Elasticsearch flush error: "
                    f"{log_error}"
                )

    def close(self):
        """Flush remaining logs on close"""
        self.flush()
        super().close()
