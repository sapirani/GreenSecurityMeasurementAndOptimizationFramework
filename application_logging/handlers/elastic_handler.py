import time
from logging import Handler
from typing import Optional, Dict, Any
from application_logging.handlers.abstract_elastic_handler import AbstractElasticSearchHandler


def get_elastic_logging_handler(
        elastic_username: str,
        elastic_password: str,
        elastic_url: str,
        index_name: str,
        starting_time: float = time.time(),
        pipeline_name: Optional[str] = None
) -> Handler:
    try:
        return ElasticSearchLogHandler(
            elastic_username=elastic_username,
            elastic_password=elastic_password,
            elastic_url=elastic_url,
            index_name=index_name,
            start_timestamp=starting_time,
            pipeline_name=pipeline_name
        )
    except ConnectionError:
        return None


class ElasticSearchLogHandler(AbstractElasticSearchHandler):
    def _inner_emit(self, doc: Dict[str, Any]) -> None:
        self.es.index(index=self.index_name, body=doc, pipeline=self.pipeline_name)

