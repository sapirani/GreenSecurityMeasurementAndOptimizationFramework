from pydantic import BaseModel, ConfigDict


class ElasticsearchConnectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_timeout: int = 10
    max_retries: int = 0
    retry_backoff_base: int = 2
    retry_backoff_cap: int = 600
