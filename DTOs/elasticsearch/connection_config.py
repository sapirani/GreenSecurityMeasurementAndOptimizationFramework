from pydantic import BaseModel, ConfigDict


class ElasticsearchConnectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_timeout: int = 10,
    max_retries: int = 0,
    initial_backoff: int = 2,
    max_backoff: int = 600,
